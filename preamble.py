import os
c_dir=os.getcwd()
#os.chdir(os.path.expanduser("~"))
if os.path.isdir('gprMax'):
    print('Hey, gprMax exists but need to be added to this session. Just a moment ...')
    os.chdir('gprMax')
    os.system("python setup.py install")
else:
    print('Hey, need to clone, compile and install gprMax. It can take a while ...')
    os.system("git clone https://github.com/gprMax/gprMax.git gprMax")
    os.chdir('gprMax')
    os.system("python setup.py build")
    os.system("python setup.py install")
    

os.chdir(c_dir)
    
#!pip install -q gprMax
print('Installing vtk')
os.system("pip install -q vtk")
os.system("pip install -q bitstruct==3.10.0")
print('Installing bitstruct')
os.system("pip install -q bitstring==3.1.5")
print('Installing bitstring')

from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

#%matplotlib inline
from IPython.core.display import display, HTML
from IPython.display import clear_output
display(HTML("<style>div.output_scroll  {height: 30em}; </style>"))
import ipywidgets as wd
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from xml.etree import ElementTree as ET

plt.rcParams["figure.figsize"] = (15,10)

print("Installing gprMax_model()")
def gprMax_model(filename):
    objects = []
    materials = []
    srcs_pml = []
    rxs = []
    with open(filename, 'rb') as f:       
        for line in f:
            if line.startswith(b'<Material'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                materials.append(tmp)
            elif line.startswith(b'<Sources') or line.startswith(b'<PML'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                srcs_pml.append(tmp)
            elif line.startswith(b'<Receivers'):
                line.rstrip(b'\n')
                tmp = (int(ET.fromstring(line).text), ET.fromstring(line).attrib.get('name'))
                rxs.append(tmp)
                
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vti = reader.GetOutput()
    shape = vti.GetDimensions()
    #mat_datarange = vti.GetCellData().GetArray('Material').GetRange()
    #print(mat_datarange)
    extent = vti.GetExtent()
    spacing = vti.GetSpacing()
    
    #print(extent, spacing, shape)
    x_start = extent[0]
    x_stop = extent[1]
    y_start = extent[2]
    y_stop = extent[3]
    z_start = extent[4]
    z_stop = extent[5]
    
    dx = spacing[0]
    dy = spacing[1]
    dz = spacing[2]
    
    if z_stop == 1:
        S1 = y_stop
        d1 = dy
        S2 = x_stop
        d2 = dx
    if y_stop == 1:
        S1 = z_stop
        d1 = dz
        S2 = x_stop
        d2 = dx
    if x_stop == 1:
        S1 = z_stop
        d1 = dz
        S2 = y_stop
        d2 = dy   
    
    domain = vtk_to_numpy(vti.GetCellData().GetArray('Material'))
    domain = domain.flatten().reshape(S1, S2)
    PML_Tx = vtk_to_numpy(vti.GetCellData().GetArray('Sources_PML'))
    PML_Tx = PML_Tx.flatten().reshape(S1, S2)
    Rx = vtk_to_numpy(vti.GetCellData().GetArray('Receivers'))
    Rx = Rx.flatten().reshape(S1, S2)
   
    for i in range(len(materials)):
        tmp_mat = materials[i]
        ind = materials.index(tmp_mat)
        w = np.where(domain == tmp_mat[0])
        if w[0].size > 0:
            objects.append(tmp_mat[1])
            domain[w]=objects.index(tmp_mat[1])           
            
    for i in range(len(srcs_pml)):
        tmp = srcs_pml[i]
        ind = srcs_pml.index(tmp)
        w = np.where(PML_Tx == tmp[0])
        if w[0].size > 0:
            objects.append(tmp[1])
            domain[w]=objects.index(tmp[1])
           
    for i in range(len(rxs)):
        tmp = rxs[i]
        ind = rxs.index(tmp)
        w = np.where(Rx == tmp[0])
        if w[0].size > 0:
            objects.append(tmp[1])
            domain[w]=objects.index(tmp[1])
    
    #create the domain plot
    
    #plt.imshow(PML_Tx, interpolation='nearest', origin='lower', alpha=0.1)
    #plt.imshow(Rx, interpolation='nearest', origin='lower', alpha=0.1)
    im = plt.imshow(domain, cmap=plt.get_cmap('jet'), interpolation='nearest', origin='lower', extent=[0, S2*d2, 0, S1*d1])
    
    #get unique entries in the domain  
    entries = np.unique(domain.ravel())
    # get the colour for every entry from the colormap used by imshow
    colours = [ im.cmap(im.norm(entry)) for entry in entries]
    # create a patch for every colour 
    patches = [ mpatches.Patch(color=colours[i], label="{l}".format(l=objects[i])) for i in range(len(entries)) ]
         
    #put patches as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    ax=plt.gca()  # get the axis
    ax.set_xlabel('Metres [m]')
    ax.set_ylabel('Metres [m]')
    #finally show the plot
    plt.show()

print("Installing gprMax_Ascan()")    
def gprMax_Ascan(filename, rxnumber, rxcomponent):
    import h5py
    """Gets A-scan output data from a model.
    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.
    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())
    g = f[path]
    pos=np.array(g.attrs['Position'])
           
    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise CmdInputError('{} output requested, but the available output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    time = np.linspace(0,outputdata.size*dt,outputdata.size)/1e-9
    f.close()

    return outputdata, time, pos

print("Installing gprMax_Bscan()")
def gprMax_Bscan(filename, rx, rxcomponent):
    import h5py
    import os
    import glob
    
    filename = filename[0:-4]
    files = glob.glob(filename + '*.out')
    outputfiles = [filename for filename in files if '_merged' not in filename]
    modelruns = len(outputfiles)
    #print(outputfiles)
    #print(modelruns)
    
    file0 = filename + str(0 + 1) + '.out'
    out, time, pos = gprMax_Ascan(file0, rx, rxcomponent)
     
    spos = np.array(pos, ndmin=2)
    bscan=np.array(out, ndmin=2)
    bscan = bscan.T
      
    for model in range(1,modelruns):
        file = filename + str(model + 1) + '.out'
        
    #for model in range(modelruns-1):
        #out, time, pos = gprMax_Ascan(outputfiles[model], rx, rxcomponent)
        out, time, pos = gprMax_Ascan(file, rx, rxcomponent)
        out = np.array(out, ndmin=2)
        bscan = np.append(bscan,out.T, axis=1)
        pos = np.array(pos, ndmin=2)
        spos = np.append(spos,pos, axis=0)

    return bscan, time, spos


def merge_files(basefilename, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.
    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """
    import h5py
    import os
    import glob
    from gprMax._version import __version__
    
    outputfile = basefilename + '_merged.out'
    files = glob.glob(basefilename + '*.out')
    outputfiles = [filename for filename in files if '_merged' not in filename]
    modelruns = len(outputfiles)
    print(modelruns)

    # Combined output file
    fout = h5py.File(outputfile, 'w')

    # Add positional data for rxs
    for model in range(modelruns):
        fin = h5py.File(basefilename + str(model + 1) + '.out', 'r')
        nrx = fin.attrs['nrx']
      
        # Write properties for merged file on first iteration
        if model == 0:
            fout.attrs['Title'] = fin.attrs['Title']
            fout.attrs['gprMax'] = __version__
            fout.attrs['Iterations'] = fin.attrs['Iterations']
            fout.attrs['dt'] = fin.attrs['dt']
            fout.attrs['nrx'] = fin.attrs['nrx']
            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx)
                grp = fout.create_group(path)
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(output, (fout.attrs['Iterations'], modelruns), dtype=fin[path + '/' + output].dtype)

        # For all receivers
        for rx in range(1, nrx + 1):
            path = '/rxs/rx' + str(rx) + '/'
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[path + '/' + output][:, model] = fin[path + '/' + output][:]

        fin.close()

    fout.close()

    if removefiles:
        for model in range(modelruns):
            file = basefilename + str(model + 1) + '.out'
            os.remove(file)
print("Installing plot_Ascan()")
def plot_Ascan(x, y):
        offset = 0
        p = plt.plot(x,y,'k-')
        plt.fill_betweenx(y,offset,x,where=(x>offset),color='k')
        plt.tight_layout(True)
        ax=plt.gca()  # get the axis
        xmin = np.min(x)
        xmax = np.max(x)
        
        ymin = np.min(y)
        ymax = np.max(y)
        
        ax.set_xlim([-np.max(np.abs(x)), np.max(np.abs(x))])
        ax.set_ylim([ymin, ymax])
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()   
        scale_str = ax.get_yaxis().get_scale()
       
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ax.set_ylabel('Time [ns]')
        ax.set_xlabel('Field Strength [V/m]')
        aspect = 0.5
        if scale_str=='linear':
            asp = abs((xmax-xmin)/(ymax-ymin))/aspect
        elif scale_str=='log':
            asp = abs((scipy.log(xmax)-scipy.log(xmin))/(scipy.log(ymax)-scipy.log(ymin)))/aspect
        ax.grid(which='both', axis='both', linestyle='-.')
        ax.set_aspect(asp)
        plt.show()

print("Installing plot_Bscan()")
def plot_Bscan(scan,time,time_offset=0):
    scan_max = np.max(np.max(np.abs(scan)))
    plt.imshow(scan, cmap='seismic', extent=[0,scan.shape[1],np.max(time)-time_offset,0-time_offset], aspect=15, vmin=-scan_max, vmax=scan_max)
    plt.colorbar
    ax=plt.gca()  # get the axis
    ax.set_xlabel('Trace Number ')
    ax.set_ylabel('Time [ns]')
    plt.show()
    

print("Installing create_model()")    
class create_model():
    
    def __init__(self,file):
    
        self.filename = file
        self.noMaterials=0
        self.noObjects = 0

        self.matAll=[]; self.objAll =[]; self.widgetList=[]; self.objectList=[]; self.wfList=[]; self.wfAll=[]; 
        self.srcAll=[]; self.rcAll=[]; self.goList=[]; self.goAll=[]; self.rem=[]

        self.accNames = ['Space', 'Materials', 'Objects', 'Source/Receiver', 'Geometry Output']
        self.add_mat = wd.Button(
                value=False,
                description='Add material',
                disabled=False)
        self.widgetList=[self.add_mat, wd.Label('free_space : Built-in identifier for air', 
                                layout=wd.Layout(width='30%', margin='-35px 0px 0px 200px')),
                        wd.Label('pec : Built-in identifier for a perfect electric conductor', 
                                layout=wd.Layout(width='50%', margin='-10px 0px 10px 200px'))]

        self.add_obj = wd.Button(
                value=False,
                description='Add object',
                disabled=False)
        
        self.rem_obj = wd.Button(
                value=False,
                description='Remove last object',
                disabled=False)
        
        self.objectList=[wd.HBox([self.add_obj, self.rem_obj],layout=wd.Layout(margin='10px 0px 20px 0px'))]

        self.wrFile = wd.Button(
                value=False,
                description='Write input file',
                disabled=False, 
                layout=wd.Layout(height='30px', width='40%', margin='10px 0px 20px 20px'))

        self.dd = [wd.Dropdown(
            options=[' ', 'Edge', 'Plate', 'Triangle', 'Box', 'Sphere', 'Cylinder'],
            description='Object {}:'.format(i+1),
            disabled=False , layout=wd.Layout(margin='20px 0px 10px -20px')) for i in range(20)]
    
        self.ddwf = wd.Dropdown(
            options=[' ', 'gaussian', 'gaussiandot', 'gaussiandotnorm', 'gaussiandotdot',
            'gaussiandotdotnorm', 'ricker', 'gaussianprime', 'gaussiandoubleprime', 'sine', 'contsine'],
            description='Waveform: ',
            disabled=False , layout=wd.Layout(width='23%', margin='2px 5px 10px 0px'))

        self.ddsrc = wd.Dropdown(
            options=[' ', 'hertzian_dipole', 'voltage_source', 'magnetic_dipole'],
            description='Source: ',
            disabled=False , layout=wd.Layout(width='23%', margin='2px 5px 10px 0px'))
 
        self.dom = [wd.Text(layout=wd.Layout(width='15%')) for i in range(3)]
        self.sp = [wd.Text(layout=wd.Layout(width='15%')) for i in range(3)]
        self.time_window = wd.Text(layout=wd.Layout(width='15%'))   
        self.title = wd.Text(layout=wd.Layout(width='50%'))
        self.msg = wd.Text(layout=wd.Layout(width='15%'))
        self.intRes = wd.Text(layout=wd.Layout(width='11%', margin='10px 0px 0px 0px'))
        self.wfAll.append([wd.Text(layout=wd.Layout(width='15%')) for i in range(3)])
        self.srcAll.append([wd.Text(layout=wd.Layout(width='13%')) for i in range(4)])
        self.rcAll.append([wd.Text(layout=wd.Layout(width='13%')) for i in range(3)])
        self.steps = [wd.Text(layout=wd.Layout(width='15%')) for i in range(6)]
        self.goAll.append([wd.Text(layout=wd.Layout(width='13%')) for i in range(9)])
        self.goAll[0].append(wd.Text(layout=wd.Layout(width='25%')))
    
        self.spaceList = [
             wd.HBox([wd.Label('Title: ', layout=wd.Layout(width='20%')), 
                      self.title]),
             wd.HBox([wd.Label('', layout=wd.Layout(width='25%')), 
                      wd.Label('X', layout=wd.Layout(width='15%')), 
                      wd.Label('Y', layout=wd.Layout(width='15%')),
                      wd.Label('Z', layout=wd.Layout(width='15%'))]),
             wd.HBox([wd.Label('Domain: ', layout=wd.Layout(width='20%')), 
                      self.dom[0], self.dom[1], self.dom[2]]),
             wd.HBox([wd.Label('', layout=wd.Layout(width='25%')), 
                      wd.Label('dx', layout=wd.Layout(width='15%')), 
             wd.Label('dy', layout=wd.Layout(width='15%')),
             wd.Label('dz', layout=wd.Layout(width='15%'))]),
             wd.HBox([wd.Label('Spacing: ', layout=wd.Layout(width='20%')), 
                      self.sp[0], self.sp[1], self.sp[2]]),
             wd.HBox([wd.Label('Time Window: ', layout=wd.Layout(width='20%')), self.time_window]),
             wd.HBox([wd.Label('Messages(y/n): ', layout=wd.Layout(width='20%')), 
                      self.msg])   
            ]

        self.propLabels=wd.HBox([wd.Label('', layout=wd.Layout(width='10%')), 
                    wd.Label('Relative Permittivity', layout=wd.Layout(width='15%')), 
                    wd.Label('\t\t\tConductivity', layout=wd.Layout(width='15%')),
                    wd.Label('Relative Permeability', layout=wd.Layout(width='15%')),
                    wd.Label('Magnetic Loss', layout=wd.Layout(width='15%')),
                    wd.Label('Name', layout=wd.Layout(width='25%'))])    
        self.srList=[wd.HBox([wd.Label(' ', layout=wd.Layout(width='12%')),
                      wd.Label('Type', layout=wd.Layout(width='11.5%')),
                      wd.Label('Scaling of max amplitude', layout=wd.Layout(width='17%')),
                      wd.Label('Center frequency', layout=wd.Layout(width='17%')) ,  
                      wd.Label('Name', layout=wd.Layout(width='5%'))], 
                      layout=wd.Layout(margin='10px 0px 10px 0px')), 
                      wd.HBox([self.ddwf, self.wfAll[0][0],self.wfAll[0][1],self.wfAll[0][2]]), 
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='12%')),
                      wd.Label('Type', layout=wd.Layout(width='11%')),
                      wd.Label('Polarization (x,y or z)', layout=wd.Layout(width='19%')),
                      wd.Label('X', layout=wd.Layout(width='12%')),
                      wd.Label('Y', layout=wd.Layout(width='13%')) ,  
                      wd.Label('Z', layout=wd.Layout(width='15%')),], 
                      layout=wd.Layout(margin='10px 0px 10px 0px')), 
                      wd.HBox([self.ddsrc, self.srcAll[0][0],self.srcAll[0][1],
                               self.srcAll[0][2],self.srcAll[0][3] ]),
                      wd.HBox([wd.Label('If voltage source specify internal resistance (Ohm):', 
                                        layout=wd.Layout(width='30%', margin='10px 0px 0px 87px')),self.intRes]),                      
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='14%')),
                      wd.Label('X', layout=wd.Layout(width='13%')),
                      wd.Label('Y', layout=wd.Layout(width='12%')),
                      wd.Label('Z', layout=wd.Layout(width='9%'))], layout=wd.Layout(margin='10px 0px 0px 0px')),
                      wd.HBox([wd.Label('Receiver: ', layout=wd.Layout(width='6.2%',margin='0px 0px 20px 25px')),
                               self.rcAll[0][0],self.rcAll[0][1],self.rcAll[0][2]]),
                      wd.HBox([wd.Label('For B-scan specify increments to move sources and receivers:', 
                                        layout=wd.Layout(width='40%', margin='10px 0px 0px 5px'))]),
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='17%')),
                      wd.Label('x', layout=wd.Layout(width='15%')),
                      wd.Label('y', layout=wd.Layout(width='15%')) ,  
                      wd.Label('z', layout=wd.Layout(width='15%'))]),
                      wd.HBox([wd.Label('Source steps: ', layout=wd.Layout(width='10%')),
                      self.steps[0],self.steps[1],self.steps[2]]),
                      wd.HBox([wd.Label('Receiver steps: ', layout=wd.Layout(width='10%')),
                      self.steps[3],self.steps[4],self.steps[5]],layout=wd.Layout(margin='0px 0px 20px 0px'))]

        self.goList=[wd.HBox([wd.Label(' ', layout=wd.Layout(width='5%')),
                      wd.Label('X start', layout=wd.Layout(width='13%')),
                      wd.Label('Y start', layout=wd.Layout(width='13%')),
                      wd.Label('Z start', layout=wd.Layout(width='13%')),
                      wd.Label('X end', layout=wd.Layout(width='13%')),
                      wd.Label('Y end', layout=wd.Layout(width='13%')),
                      wd.Label('Z end', layout=wd.Layout(width='9%'))
                ], layout=wd.Layout(margin='10px 0px 0px 0px')),
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='1%')),
                               self.goAll[0][0],self.goAll[0][1],self.goAll[0][2],
                               self.goAll[0][3],self.goAll[0][4],self.goAll[0][5]]), 
            wd.HBox([wd.Label(' ', layout=wd.Layout(width='6%')),
                      wd.Label('dx', layout=wd.Layout(width='13%')),
                      wd.Label('dy', layout=wd.Layout(width='13%')),
                      wd.Label('dz', layout=wd.Layout(width='9%'))], layout=wd.Layout(margin='10px 0px 0px 0px')),
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='1%')),
                               self.goAll[0][6],self.goAll[0][7],self.goAll[0][8]]), 
            wd.HBox([wd.Label(' ', layout=wd.Layout(width='3%')),
                      wd.Label('Geometry filename (no extension)', layout=wd.Layout(width='22%'))],
                      layout=wd.Layout(margin='10px 0px 0px 0px')),
                      wd.HBox([wd.Label(' ', layout=wd.Layout(width='1%')),
                               self.goAll[0][9]], layout=wd.Layout(margin='0px 0px 20px 0px')), self.wrFile]
    
        self.add_mat.on_click(self.on_button_clicked)
        self.add_obj.on_click(self.on_button_clicked2)
        self.rem_obj.on_click(self.on_button_clicked3)
        self.wrFile.on_click(self.write_file)
        
        [self.dd[i].observe(self.on_change,'value') for i in range(20)]
        self.gb=wd.GridBox(children=self.spaceList, layout=wd.Layout(grid_template_columns="repeat(1, 500px)"))    
        self.gb2=wd.GridBox(children=self.widgetList, layout=wd.Layout(grid_template_columns="repeat(1, 900px)"))  
        self.gb3=wd.GridBox(children=self.objectList, layout=wd.Layout(grid_template_columns="repeat(1, 1100px)"))  
        self.gb4=wd.GridBox(children=self.srList, layout=wd.Layout(grid_template_columns="repeat(1, 1000px)"))  
        self.gb5=wd.GridBox(children=self.goList, layout=wd.Layout(grid_template_columns="repeat(1, 900px)"))  

        self.acc = wd.Accordion(children=[self.gb, self.gb2, self.gb3, self.gb4, self.gb5])
        for i in range(5):
            self.acc.set_title(i,self.accNames[i])
            
        display(self.acc)

        
    def on_button_clicked(self,butt):
        
        if self.noMaterials<1:
            self.gb2.children+=(self.propLabels,)
        self.noMaterials+= 1
        noMaterials = self.noMaterials
        self.matAll.append([wd.Text(layout=wd.Layout(width='15%')) for i in range(5)])
        self.gb2.children+=(wd.HBox([wd.Label('Material {}'.format(noMaterials), layout=wd.Layout(width='9%')),
                        self.matAll[noMaterials-1][0], self.matAll[noMaterials-1][1],self.matAll[noMaterials-1][2],
                        self.matAll[noMaterials-1][3],self.matAll[noMaterials-1][4]] ),)

    def on_button_clicked2(self,butt):

        self.noObjects+=1
        self.gb3.children+=(self.dd[self.noObjects-1],)   
        
    def on_button_clicked3(self,butt):
        
        if self.noObjects>0:
            self.gb3.children = self.gb3.children[0:-self.rem[self.noObjects-1]]
            self.objAll.pop()
            self.rem.pop()
            self.dd[self.noObjects-1].disabled=False
            self.dd[self.noObjects-1].value= " "
            self.noObjects-=1
        
        

    def write_file(self,butt):
        

        if any([val.value == "" for val in self.dom]): 

            raise ValueError(' Size of model is required! ') 
            return
                           
        if any([val.value == "" for val in self.sp]): 
               
            raise ValueError(' Discretization is required! ') 
            return
            
        if self.time_window.value == "":     

            raise ValueError(' Time window is required!  ') 
            return
            
        with open(self.filename, 'w') as fid:
            
            fid.write('#title: ' + self.title.value + '\n') 
            fid.write('#domain: ' + ' '.join([str(elem.value) for elem in self.dom]) + '\n')
            fid.write('#dx_dy_dz: ' + ' '.join([str(elem.value) for elem in self.sp]) + '\n')
            fid.write('#time_window: ' + self.time_window.value + '\n')
                   
            if not self.msg.value=="":
                fid.write('#messages: ' + self.msg.value + '\n')
                
            fid.write('#num_threads: 1 \n\n')
            
            for i in range(self.noMaterials):
                fid.write('#material: ' + ' '.join([str(elem.value) for elem in self.matAll[i]]) + '\n')
                       
            if not any([val.value == "" for val in self.wfAll[0]]):

                fid.write('\n')    
                fid.write('#waveform: ' +  self.ddwf.value + ' '  +' '.join([str(elem.value) for elem in self.wfAll[0]]) + '\n')
                if self.ddsrc.value=='voltage_source':
                    fid.write('#' +  self.ddsrc.value +': '  +' '.join([str(elem.value) for elem in self.srcAll[0][0:4]]) + ' '
                     + self.intRes.value + ' '  + self.wfAll[0][-1].value + '\n')
                else:
                    fid.write('#' +  self.ddsrc.value +': '  +' '.join([str(elem.value) for elem in self.srcAll[0]]) + ' '
                      + self.wfAll[0][-1].value + '\n')
                fid.write('#rx: ' +' '.join([str(elem.value) for elem in self.rcAll[0]]) + '\n')
                fid.write('\n')
            
            if not any([val.value == "" for val in self.steps[0:3]]):
                fid.write('#src_steps: '  +' '.join([str(elem.value) for elem in self.steps[0:3]]) + '\n')
                
            if not any([val.value == "" for val in self.steps[3:6]]):
                fid.write('#rx_steps: '  +' '.join([str(elem.value) for elem in self.steps[3:6]]) + '\n')                
            
            for i in range(self.noObjects):
                fid.write('#' +  self.dd[i].value.lower() +': '  +' '.join([str(elem.value) for elem in self.objAll[i]]) + '\n')
                
            fid.write('\n')    
            if not any([val.value == "" for val in self.goAll[0]]):
                fid.write('#geometry_view: ' +' '.join([str(elem.value) for elem in self.goAll[0]]) + ' n \n')
        fid.close()
        print(' Input file ' + self.filename +' was written successfully! ')
        
    def on_change(self,change):  
   
        noObjects = self.noObjects
        if change['new'] == 'Edge' or change['new']=='Plate' or change['new']=='Box':        
            self.objAll.append([wd.Text(layout=wd.Layout(width='10%')) for i in range(7)])
        #noObjects+= 1         
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='2%')), 
                      wd.Label('X start', layout=wd.Layout(width='10%')), 
                      wd.Label('Y start', layout=wd.Layout(width='10%')),
                      wd.Label('Z start', layout=wd.Layout(width='10%')) ,            
                      wd.Label('X end', layout=wd.Layout(width='10%')), 
                      wd.Label('Y end', layout=wd.Layout(width='10%')),
                      wd.Label('Z end', layout=wd.Layout(width='10%')),
                      wd.Label('Material', layout=wd.Layout(width='10%'))], 
                               layout=wd.Layout(margin='10px 0px 0px 0px')) ,)    
            self.gb3.children+=(wd.HBox([self.objAll[noObjects-1][0], self.objAll[noObjects-1][1],
                                self.objAll[noObjects-1][2], self.objAll[noObjects-1][3], 
                                self.objAll[noObjects-1][4],self.objAll[noObjects-1][5],
                                self.objAll[noObjects-1][6]], layout=wd.Layout(margin='0px 0px 10px 0px')), )
            self.dd[noObjects-1].disabled=True
            self.rem.append(3)
        elif change['new'] == 'Triangle':
            self.objAll.append([wd.Text(layout=wd.Layout(width='7%')) for i in range(11)])      
        
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='10%')),
                      wd.Label('Apex 1', layout=wd.Layout(width='20%')), 
                      wd.Label('Apex 2', layout=wd.Layout(width='20%')),
                      wd.Label('Apex 3', layout=wd.Layout(width='20%'))], 
                         layout=wd.Layout(margin='10px 0px 0px 0px')), )     
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='2%')), 
                      wd.Label('X', layout=wd.Layout(width='7%')), 
                      wd.Label('Y', layout=wd.Layout(width='7%')),
                      wd.Label('Z', layout=wd.Layout(width='7%')) ,            
                      wd.Label('X', layout=wd.Layout(width='7%')), 
                      wd.Label('Y', layout=wd.Layout(width='7%')),
                      wd.Label('Z', layout=wd.Layout(width='7%')),
                      wd.Label('X', layout=wd.Layout(width='7%')), 
                      wd.Label('Y', layout=wd.Layout(width='7%')),
                      wd.Label('Z', layout=wd.Layout(width='5%')),
                      wd.Label('Thickness', layout=wd.Layout(width='7%')),
                      wd.Label('Material', layout=wd.Layout(width='8%'))]) ,)    
            self.gb3.children+=(wd.HBox([self.objAll[noObjects-1][0], self.objAll[noObjects-1][1],
                                         self.objAll[noObjects-1][2], self.objAll[noObjects-1][3], 
                                         self.objAll[noObjects-1][4], self.objAll[noObjects-1][5],
                                         self.objAll[noObjects-1][6], self.objAll[noObjects-1][7],
                                         self.objAll[noObjects-1][8], self.objAll[noObjects-1][9],
                                         self.objAll[noObjects-1][10]], layout=wd.Layout(margin='0px 0px 10px 0px')), )
            self.dd[noObjects-1].disabled=True
            self.rem.append(4)
        
        elif change['new'] == 'Sphere':
            self.objAll.append([wd.Text(layout=wd.Layout(width='10%')) for i in range(5)])      
        
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='10%')),
                      wd.Label('Center of the sphere', layout=wd.Layout(width='20%')),], 
                      layout=wd.Layout(margin='10px 0px 0px 0px')), )     
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='2%')), 
                      wd.Label('X', layout=wd.Layout(width='10%')), 
                      wd.Label('Y', layout=wd.Layout(width='10%')),
                      wd.Label('Z', layout=wd.Layout(width='10%')) ,            
                      wd.Label('Radius', layout=wd.Layout(width='10%')), 
                      wd.Label('Material', layout=wd.Layout(width='10%'))]), )
            self.gb3.children+=(wd.HBox([self.objAll[noObjects-1][0], self.objAll[noObjects-1][1],
                                         self.objAll[noObjects-1][2], self.objAll[noObjects-1][3], 
                                         self.objAll[noObjects-1][4]], layout=wd.Layout(margin='0px 0px 10px 0px')), )
            self.dd[noObjects-1].disabled=True
            self.rem.append(4)
        
        elif change['new'] == 'Cylinder':
            self.objAll.append([wd.Text(layout=wd.Layout(width='10%')) for i in range(8)])      
        
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='10%')),
                      wd.Label('Center of 1st face', layout=wd.Layout(width='30%')),
                      wd.Label('Center of 2nd face', layout=wd.Layout(width='20%'))       ], 
                      layout=wd.Layout(margin='10px 0px 0px 0px')), )     
            self.gb3.children+=(wd.HBox([wd.Label(' ', layout=wd.Layout(width='2%')), 
                      wd.Label('X', layout=wd.Layout(width='10%')), 
                      wd.Label('Y', layout=wd.Layout(width='10%')),
                      wd.Label('Z', layout=wd.Layout(width='10%')) ,            
                      wd.Label('X', layout=wd.Layout(width='10%')), 
                      wd.Label('Y', layout=wd.Layout(width='10%')),
                      wd.Label('Z', layout=wd.Layout(width='10%')),     
                      wd.Label('Radius', layout=wd.Layout(width='10%')),
                      wd.Label('Material', layout=wd.Layout(width='10%')),
                               ]), )
            self.gb3.children+=(wd.HBox([self.objAll[noObjects-1][0], self.objAll[noObjects-1][1],
                                         self.objAll[noObjects-1][2], self.objAll[noObjects-1][3], 
                                         self.objAll[noObjects-1][4], self.objAll[noObjects-1][5],
                                         self.objAll[noObjects-1][6], self.objAll[noObjects-1][7]], 
                                         layout=wd.Layout(margin='0px 0px 10px 0px')), )
            self.dd[noObjects-1].disabled=True   
            self.rem.append(4)
            


print("Installing view_file()")
def view_file(filename):
    f = open(filename, 'r')
    inputFile = f.read()
    print(inputFile)

    
print("Installing gprMax_to_dzt()")    
def gprMax_to_dzt(filename, rx, rxcomponent, centerFreq, distTx_Rx, trace_step):
    
    import h5py as h5
    import os
    import sys
    import struct
    import bitstruct
    import datetime
    from bitstring import Bits
    from scipy import signal
    
    # ------------------------------- Information specified by the user ---------------------------------------

    # Specify gprMax file path name
    file_path_name = filename

    # Specify center frequency (MHz)
    center_freq = centerFreq

    # Specify Tx-Rx distance
    distance = distTx_Rx

    # Trace step
    trace_step = trace_step

    # Choose E-field component
    comp = rxcomponent

    # ---------------------------------------------------------------------------------------------------------    
        
    # Read gprMax data
    
    bscan, _, _ = gprMax_Bscan(filename+'.out', rx, rxcomponent)   
    data = np.array(bscan)
    
    # Read time step
    #file = h5.File(filename[0:-4]+'1.out', 'r')
    file = h5.File(filename+'1.out', 'r')
    time_step = file.attrs['dt']
    file.close()

    data = (data * 32767)/ np.max(np.abs(data))
    data[data > 32767] = 32767
    data[data < -32768] = -32768
    data = np.round(data)

    # Number of samples and traces
    [noSamples, noTraces] = np.shape(data)

    # Convert time step to ns
    time_step = time_step*10**9

    # Sampling frequency (MHz)
    sampling_freq = (1 / time_step)*10**3

    # Time window (ns)
    time_window = time_step*noSamples

    # DZT file name
    fileName = filename

    # Resample data to 1024 samples

    data = signal.resample(data, 1024)
    time_step = time_window / np.shape(data)[0]
    sampling_freq = (1 / time_step)*10**3

    # ------------------------------------------------ DZT file header -----------------------------------------------------


    tag = 255                                      # 0x00ff if header, 0xfnff for old file Header
    dataOffset = 1024                              # Constant 1024
    noSamples = np.shape(data)[0]                  # Number of samples
    bits = 16                                      # Bits per data word (8 or 16)
    binaryOffset = 32768                           # Binary offset (8 bit -> 128, 16 bit -> 32768)
    sps = 0                                        # Scans per second
    spm = 1 / trace_step                           # Scans per metre
    mpm = 0                                        # Meters per mark
    position = 0                                   # Position (ns)
    time_window = time_window                      # Time window (ns)
    noScans = 0                                    # Number of passes for 2D files

    dateTime = datetime.datetime.now()             # Current datetime

    # Date and time created
    createdSec = dateTime.second
    if createdSec > 29: createdSec = 29
    createdMin = dateTime.minute
    createdHour = dateTime.hour
    createdDay = dateTime.day
    createdMonth = dateTime.month
    createdYear = dateTime.year-1980

    # Date and time modified
    modifiedSec = dateTime.second
    if modifiedSec > 29: modifiedSec = 29
    modifiedMin = dateTime.minute
    modifiedHour = dateTime.hour
    modifiedDay = dateTime.day
    modifiedMonth = dateTime.month
    modifiedYear = dateTime.year-1980

    offsetRG = 0                                   # Offset to range gain function
    sizeRG = 0                                     # Size of range gain function
    offsetText = 0                                 # Offset to text
    sizeText = 0                                   # Size of text
    offsetPH = 0                                   # Offset to processing history
    sizePH = 0                                     # Size of processing history
    noChannels = 1                                 # Number of channels
    epsr = 5                                       # Average dielectric constant
    topPosition = 0                                # Top position (m)
    vel = (299792458 / np.sqrt(epsr)) * 10 ** -9
    range0 = vel * (time_window / 2)                # Range (meters)
    xStart = 0                                     # X start coordinate
    xFinish = noTraces*trace_step-trace_step       # X finish coordinate
    servoLevel = 0                                 # Gain servo level
    reserved = 0                                   # Reserved
    antConfig = 0                                  # Antenna Configuration
    setupConfig = 0                                # Setup Configuration
    spp = 0                                        # Scans per pass
    noLine = 0                                     # Line number
    yStart = 0                                     # Y start coordinate
    yFinish = 0                                    # Y finish coordinate
    lineOrder = 0
    dataType = 2                                   # Data type

    antennaName ='antName'
    if len(antennaName) > 14:
        antennaName = antennaName[0:14]
    elif len(antennaName) < 14:
        antennaName = antennaName.ljust(14)

    channelMask= 0                                 # Channel mask

    fName = fileName                               # File name
    if len(fName) > 12:
        fName = fName[0:12]
    elif len(fName) < 12:
        fName = fName.ljust(12)

    checkSum = 0                                   # Check sum for header

    # -------------------------------------------------------------------------------------------------------------------


    # ----------------------------------------- Convert to bytes and write to file --------------------------------------

    # Open file to write

    with open(fileName + '.dzt', 'wb') as fid:
 
        # Write header

        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, tag); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, dataOffset); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, noSamples); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, bits); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('s16<', dataStruct, 0, binaryOffset); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, sps); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, spm); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, mpm); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, position); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, time_window); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, noScans); fid.write(dataStruct); 

        sec = Bits(uint=createdSec, length=5)
        min = Bits(uint=createdMin, length=6)
        hour = Bits(uint=createdHour, length=5)
        day = Bits(uint=createdDay, length=5)
        month = Bits(uint=createdMonth, length=4)
        year = Bits(uint=createdYear, length=7)
        b = Bits().join([year, month, day, hour, min, sec])
        createDate = b.tobytes(); fid.write(bitstruct.pack('>r32<', createDate))
    
        sec = Bits(uint=modifiedSec, length=5)
        min = Bits(uint=modifiedMin, length=6)
        hour = Bits(uint=modifiedHour, length=5)
        day = Bits(uint=modifiedDay, length=5)
        month = Bits(uint=modifiedMonth, length=4)
        year = Bits(uint=modifiedYear, length=7)
        b = Bits().join([year, month, day, hour, min, sec])
        modifiedDate = b.tobytes(); fid.write(bitstruct.pack('>r32<', modifiedDate))
    
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, offsetRG); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, sizeRG); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, offsetText); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, sizeText); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, offsetPH); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, sizePH); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, noChannels); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, epsr); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, topPosition); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, range0); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, xStart); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, xFinish); fid.write(dataStruct);
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, servoLevel); fid.write(dataStruct); 
        dataStruct = bytearray(3); bitstruct.pack_into('r24<', dataStruct, 0, reserved); fid.write(dataStruct);
        dataStruct = bytearray(1); bitstruct.pack_into('u8<', dataStruct, 0, antConfig); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, setupConfig); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, spp); fid.write(dataStruct); 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, noLine); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, yStart); fid.write(dataStruct); 
        dataStruct = bytearray(4); bitstruct.pack_into('f32<', dataStruct, 0, yFinish); fid.write(dataStruct); 
        dataStruct = bytearray(1); bitstruct.pack_into('u8<', dataStruct, 0, lineOrder); fid.write(dataStruct); 
        dataStruct = bytearray(1); bitstruct.pack_into('r8<', dataStruct, 0, dataType); fid.write(dataStruct);
        fid.write(bitstruct.pack('t14<', antennaName)) 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, channelMask); fid.write(dataStruct);
        fid.write(bitstruct.pack('t12<', fName)) 
        dataStruct = bytearray(2); bitstruct.pack_into('u16<', dataStruct, 0, checkSum); fid.write(dataStruct); 

        # Move to 1024 to write data

        fid.seek(dataOffset)
        data = data + binaryOffset
        data = np.array(data,dtype='<H')
        fid.write(data.T.astype('<H').tobytes());

        # Close file 
    
        fid.close()
    
    print('Dzt file has been written!')
    