#! /usr/bin/env python
import numpy as np
from  matplotlib import pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from scipy.spatial import cKDTree

filename = 'phot'
filters   = np.array(['Z087','Y106','J129','H158','F184'])
AB_Vega = np.array([0.487,0.653,0.958,1.287,1.552])
tol=1

def DoAll():
    data = [ascii.read(filters[k]+'_stips.txt',format='ipac')\
            for k in range(len(filters))]
    xy = read_ascii_col(filename,'xy',2)
    vega_mags,mag_errors = [read_ascii_col(filename,\
                            k,len(filters)) for k in ['mags','errs']]
    SNR,Crowd,Round,Sharp,Sky= [read_ascii_col(filename,\
                            k,len(filters)) for k in ['snr','crowd','round','sharp','sky']]
    
    all_in = lambda a,b: input_sources(data,a,b)
    all_out = lambda a,b: output_sources(xy,vega_mags,mag_errors,\
                                         SNR,Crowd,Round,Sharp,Sky,a,b)
    t1,t2 = all_in(0,3),all_out(0,3) # only Z and H
    
    c1_in,c2_in,X,Y,typ_in = t1['c1_in'],t1['c2_in'],t1['X'],t1['Y'],t1['typ_in']
    err,rnd,shr,crd,snr,sky = t2['err'],t2['rnd'],t2['shr'],t2['crd'],t2['snr'],t2['sky']
    x,y,m1,m2 = t2['xy'][0],t2['xy'][1],t2['mag'][0],t2['mag'][1]

    tt = typ_in=='point'
    c1_in,c2_in,X,Y,typ_in = c1_in[tt],c2_in[tt],X[tt],Y[tt],typ_in[tt]; del tt
    
    tt = (err[0]<0.3)&(err[1]<0.3)&\
          (snr[0]>1)&(snr[0]<500)&(snr[1]>1)&(snr[1]<500)&\
           (shr[0]>-0.2)&(shr[0]<0.2)&(shr[1]>-0.4)&(shr[1]<0.4)&\
            (rnd[0]>-10)&(rnd[0]<15)&(rnd[1]>-10)&(rnd[1]<15)&\
             (crd[0]<0.1)&(crd[1]<0.4)
    x,y,m1,m2,snr1,snr2,sky1,sky2 = x[tt],y[tt],\
        m1[tt],m2[tt],snr[0][tt],snr[1][tt],sky[0][tt],sky[1][tt]; del tt

    in1, typ_out = match_in_out(tol,X,Y,x,y,typ_in)

    tt = typ_out=='point'
    x,y,snr1,snr2,sky1,sky2,typ_out = x[tt],y[tt],snr1[tt],snr2[tt],sky1[tt],sky2[tt],typ_out[tt]; del tt

    tt = in1!=-1
    in1,X,Y,c1_in,c2_in,typ_in = in1[tt],X[tt],Y[tt],c1_in[tt],c2_in[tt],typ_in[tt]; del tt

    in1, temp = match_in_out(tol,X,Y,x,y,typ_in)
    
    snr_out1 = [snr1[i] for i in in1]
    snr_out2 = [snr2[i] for i in in1]
    sky_out1 = [sky1[i] for i in in1]
    sky_out2 = [sky2[i] for i in in1]

    plot_xy(c1_in,snr_out1,xlabel='Count Rate (input)',ylabel='SNR (recovered)',title='WFI Z087, 10x10',xlim1=0,xlim2=10,ylim1=0,ylim2=310,outfile='Z087_10_10',fmt='png',n=4)
    plot_xy(c2_in,snr_out2,xlabel='Count Rate (input)',ylabel='SNR (recovered)',title='WFI H158, 10x10',xlim1=0,xlim2=10,ylim1=0,ylim2=110,outfile='H158_10_10',fmt='png',n=4)
    
    tab = [c1_in,snr_out1,sky_out1,c2_in,snr_out2,sky_out2]
    nms = ('Z087_Countrate','Z087_SNR','Z087_Sky','H158_Countrate','H158_SNR','H158_Sky')
    fmt = {'Z087_Countrate':'%12.8f','Z087_SNR':'%10.5f','Z087_Sky':'%10.5f','H158_Countrate':'%12.8f','H158_SNR':'%10.5f','H158_Sky':'%10.5f'}
    t   = Table(tab, names=nms)
    ascii.write(t, 'SNR_Count_10_10.txt', format='fixed_width', delimiter=' ', formats=fmt)
    
    return None

'''Match input to recovered and label recovered'''
def match_in_out(tol,X,Y,x,y,typ_in):
    in1 = matchLists(tol,X,Y,x,y)
    in2 = in1!=-1
    in3 = in1[in2]
    in4 = np.arange(len(x))
    in5 = np.setdiff1d(in4,in3)
    typ_out = np.empty(len(x),dtype='<U10')
    typ_out[in3] = typ_in[in2]
    typ_out[in5] = 'other'
    return in1, typ_out


''' Quick match; returns index of 2nd list coresponding to position in 1st '''
def matchLists(tol,x1,y1,x2,y2):
    d1 = np.empty((x1.size, 2))
    d2 = np.empty((x2.size, 2))
    d1[:,0],d1[:,1] = x1,y1
    d2[:,0],d2[:,1] = x2,y2
    t = cKDTree(d2)
    tmp, in1 = t.query(d1, distance_upper_bound=tol)
    in1[in1==x2.size] = -1
    return in1



'''Pick sources added in both bands as same object types'''
def input_sources(data,i,j):
    m1_in,m2_in,c1_in,c2_in,X1,Y1,X2,Y2 = data[i]['vegamag'],data[j+1]['vegamag'],\
        data[i]['countrate'],data[j+1]['countrate'],\
        data[i]['x'],data[i]['y'],data[j+1]['x'],data[j+1]['y']
    typ1_in, typ2_in = data[i]['type'], data[j+1]['type']
    in12 = matchLists(0.1,X1,Y1,X2,Y2)
    m1_in,c1_in,X1,Y1,typ1_in = m1_in[in12!=-1],c1_in[in12!=-1],X1[in12!=-1],Y1[in12!=-1],typ1_in[in12!=-1]
    in12 = in12[in12!=-1]
    m2_in,c2_in,X2,Y2,typ2_in = m2_in[in12],c2_in[in12],X2[in12],Y2[in12],typ2_in[in12]
    tt = typ1_in==typ2_in
    m1_in,m2_in,c1_in,c2_in,X,Y,typ_in = m1_in[tt],m2_in[tt],c1_in[tt],c2_in[tt],X1[tt],Y1[tt],typ1_in[tt]
    return dict(zip(['m1_in','m2_in','c1_in','c2_in','X','Y','typ_in'],[m1_in,m2_in,c1_in,c2_in,X,Y,typ_in]))


'''Recovered source photometry and quality params'''
def output_sources(xy,mags,errors,SNR,Crowd,Round,Sharp,Sky,i,j):
    nms = ['xy','mag','err','snr','crd','rnd','shr','sky']
    K = [[xy[0],xy[1]],[mags[i],mags[j+1]],[errors[i],errors[j+1]],[SNR[i],SNR[j+1]],\
         [Crowd[i],Crowd[j+1]],[Round[i],Round[j+1]],[Sharp[i],Sharp[j+1]],[Sky[i],Sky[j+1]]]
    return dict(zip(nms,K))



'''Simple Plotting'''
def plot_xy(x,y,xlabel='',ylabel='',title='',stars=[],other=[],\
              xlim1=-1,xlim2=1,ylim1=-7.5,ylim2=7.5,\
              fileroot='',outfile='test',fmt='png',n=4):
    plt.rc("font", family='serif', weight='bold')
    plt.rc("xtick", labelsize=15); plt.rc("ytick", labelsize=15)
    fig = plt.figure(1, ((10,10)))
    fig.suptitle(title,fontsize=5*n)
    if not len(x[other]):
        plt.plot(x, y,'k.',markersize=1,alpha=0.5)
    else:
        plt.plot(x[stars],y[stars],'b.',markersize=2,\
            alpha=0.5,zorder=2,label='Stars: %d' % len(x[stars]))
        plt.plot(x[other],y[other],'k.',markersize=1,\
            alpha=0.75,zorder=1,label='Other: %d' % len(x[other]))
        plt.legend(loc=4,fontsize=20)
    plt.xlim(xlim1,xlim2); plt.ylim(ylim1,ylim2)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.savefig(fileroot+outfile+'.'+str(fmt))
    return plt.close()




'''Return requested columns in a NumPy array'''
def read_ascii_col(filename,suff,ncol):
    if isinstance(ncol,int):
        ncol = range(ncol)
    _tmp, _file = [], '.'.join((filename,suff))
    _data = ascii.read(_file)
    for i in ncol:
        _tmp.append(_data['col'+str(i+1)])
    return np.array(_tmp)


'''If executed from command line'''
if __name__ == '__main__':
    DoAll()
    
