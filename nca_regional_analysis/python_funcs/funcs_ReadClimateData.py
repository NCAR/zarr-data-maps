import xarray as xr
import numpy as np
import funcs_general as fg
import funcs_pranalysis as fpr
import pickle


def read_CESM2_xpr():
    with open('/glade/derecho/scratch/nlybarger/data/CESM2-LE/cesm2.xprvars.pkl', 'rb') as file:
        nci = pickle.load(file)
    tplatmid = 38.83
    tplonmid = 253.5
    tplatind = np.abs(nci['hist']['PRECT'].lat - tplatmid).argmin().values
    tplonind = np.abs(nci['hist']['PRECT'].lon - tplonmid).argmin().values
    tplatindslic = slice(tplatind-1,tplatind+2)
    tplonindslic = slice(tplonind-1,tplonind+2)

    lvlatmid = 39.2
    lvlonmid = 240.5
    lvlatind = np.abs(nci['hist']['PRECT'].lat - lvlatmid).argmin().values
    lvlonind = np.abs(nci['hist']['PRECT'].lon - lvlonmid).argmin().values
    lvlatindslic = slice(lvlatind-1,lvlatind+2)
    lvlonindslic = slice(lvlonind-1,lvlonind+2)

    nDayAccum=[1,3,7,15]
    MinDistDD=7
    tpmonlist = np.arange(4,11)
    lvmonlist = [11,12,1,2,3,4,5,6,7]

    # Number of extremes determined by:
    # Dividing total number of days into the maximum number of precipitation events
    # Maximum events is given by the number of accumulation days + the minimum distance
    # This is then multiplied by topPercentile to take only the top x% of events
    tp_xtensi = {}
    tp_xtdati = {}
    lv_xtensi = {}
    lv_xtdati = {}

    tp_xtensi_filt = {}
    tp_xtdati_filt = {}
    tp_datstrt_filt = {}
    tp_datend_filt = {}
    for exp in ['hist','ssp370']:
        print('Determining extreme precipitation days for experiment: ' + exp)
        lv_xtensi[exp] = {}
        lv_xtdati[exp] = {}
        tp_xtensi[exp] = {}
        tp_xtdati[exp] = {}


        tp_xtensi_filt[exp] = {}
        tp_xtdati_filt[exp] = {}
        tp_datstrt_filt[exp] = {}
        tp_datend_filt[exp] = {}
        for nd in nDayAccum:
            print('|===| for ' + str(nd) + ' day accumulations')
            lv_topPercentile = 0.1
            if nd == 1:
                tp_topPercentile = 0.4
            elif nd == 15:
                tp_topPercentile = 0.1
            else:
                tp_topPercentile = 0.2
                
            nday = len(nci[exp]['PRECT']['time'])
            tpnxt = int((nday*(len(tpmonlist)/12)/(nd+MinDistDD))*tp_topPercentile)*10
            lvnxt = int((nday*(len(lvmonlist)/12)/(nd+MinDistDD))*lv_topPercentile)*10
            if tpnxt%2 == 0:
                tpnxt = tpnxt-1
            if lvnxt%2 == 0:
                lvnxt = lvnxt-1
            # print('|===| over Taylor Park')
            # print('|===|===| Accumulation Period: ' + str(nd) + ' Day')
            # print('|===|===|===| Number of Extreme Events selected: '+str(tpnxt))
            if nDayAccum == 1:
                tp_xtensi[exp][nd],tp_xtdati[exp][nd],tp_xtensi_filt[exp][nd],tp_xtdati_filt[exp][nd],tp_datstrt_filt[exp][nd],tp_datend_filt[exp][nd] = \
                fpr.find_pr_extremes(tplatindslic,tplatind,tplonindslic,tplonind,nci[exp],'TaylorPark',
                exp,monlist=tpmonlist,nDayAccum=nd,MinDistDD=MinDistDD,nxt=tpnxt,topPercentile=tp_topPercentile)
            else:
                tp_xtensi[exp][nd],tp_xtdati[exp][nd],tp_xtensi_filt[exp][nd],tp_xtdati_filt[exp][nd],tp_datstrt_filt[exp][nd],tp_datend_filt[exp][nd] = \
                fpr.find_pr_extremes(tplatindslic,tplatind,tplonindslic,tplonind,nci[exp],'TaylorPark',
                exp,monlist=tpmonlist,nDayAccum=nd,MinDistDD=MinDistDD,nxt=tpnxt,topPercentile=tp_topPercentile)
            # print('|===| over Lahontan Valley')
            # print('|===|===| Accumulation Period: ' + str(nd) + ' Day')
            # print('|===|===|===| Number of Extreme Events selected: '+str(lvnxt))
            lv_xtensi[exp][nd],lv_xtdati[exp][nd],_,_,_,_ = \
            fpr.find_pr_extremes(lvlatindslic,lvlatind,lvlonindslic,lvlonind,nci[exp],'LahontanValley',
            exp,monlist=lvmonlist,nDayAccum=nd,MinDistDD=MinDistDD,nxt=lvnxt,topPercentile=lv_topPercentile)
    tppr = {}
    tppr['hist'] = nci['hist']['PRECT'].isel(lat=tplatindslic,lon=tplonindslic).sel(time=nci['hist']['PRECT']['time.month'].isin(tpmonlist)).copy()
    tppr['ssp370'] = nci['ssp370']['PRECT'].isel(lat=tplatindslic,lon=tplonindslic).sel(time=nci['ssp370']['PRECT']['time.month'].isin(tpmonlist)).copy()

    lvpr = {}
    lvpr['hist'] = nci['hist']['PRECT'].isel(lat=lvlatindslic,lon=lvlonindslic).sel(time=nci['hist']['PRECT']['time.month'].isin(lvmonlist)).copy()
    lvpr['ssp370'] = nci['ssp370']['PRECT'].isel(lat=lvlatindslic,lon=lvlonindslic).sel(time=nci['ssp370']['PRECT']['time.month'].isin(lvmonlist)).copy()

    return tp_xtdati_filt, tp_xtensi_filt, lv_xtdati, lv_xtensi, tppr, lvpr, nci


def standardize_CESM2(xtdati, xtensi, nci, domainname):
    exps = list(xtdati.keys())
    accums = list(xtdati[exps[0]].keys())
    nlat = nci['hist']['PRECT']['lat'].size
    nlon = nci['hist']['PRECT']['lon'].size
    for exp in exps:
        for accum in accums:
            nrank = len(xtensi[exp][accum])
            dat = np.zeros((nrank, nlat, nlon))
            
            for irank in range(nrank):
                if accum == 1:
                    dat[irank, :, :] = nci[exp]['PRECT']['PRECT'].isel(time=xtdati[exp][accum][irank], ens=xtensi[exp][accum][irank])
                else:
                    bna = int(accum/2)
                    timslic = slice(xtdati[exp][accum][irank]-bna, xtdati[exp][accum][irank]+bna)
                    dat[irank, :, :] = nci[exp]['PRECT']['PRECT'].isel(time=timslic, ens=xtensi[exp][accum][irank]).sum(dim='time')

            datout = xr.Dataset(
                {
                    'pr': (['rank', 'lat', 'lon'], dat)
                },
                coords={
                    'rank': np.arange(1, nrank+1), 
                    'lat': nci[exp]['PRECT']['lat'].data, 
                    'lon': nci[exp]['PRECT']['lon'].data
                },
            )
            datout['pr'].attrs['units'] = 'mm'
            datout['pr'].attrs['long_name'] = 'Precipitation'
            datout.to_netcdf('/glade/derecho/scratch/nlybarger/data/CESM2-LE/CESM2.PRECT.' + domainname + '_' + exp + '_' + str(accum) + '.accum.nc')
            del dat
            del datout
    # pickle.save(datout, '/glade/derecho/scratch/nlybarger/data/CESM2-LE/cesm2.xprvars.standardized.pkl')
    # return datout
            

def standardize_GARD(datin, xprtxt, accum, domainname, experiment):

    nlat = datin['lat'].size
    nlon = datin['lon'].size

    eventnum = xprtxt[:, 0]
    eventstrt = xprtxt[:, 1]
    eventend = xprtxt[:, 2]
    eventens = xprtxt[:, 3]

    if accum == 1:
        eventlen = 600
        accumstr = '1.accum'
    elif accum == 3:
        eventlen = 400
        accumstr = '3.accum'
    elif accum == 7:
        eventlen = 250
        accumstr = '7.accum'
    elif accum == 15:
        eventlen = 100
        accumstr = '15.accum'
    
    np_datout = np.zeros((eventlen, nlat, nlon))
    for i in range(eventlen):
        timslic = slice(str(eventstrt[i]), str(eventend[i]))
        np_datout[i,:,:] = datin['pcp'].sel(event=i+1, time=timslic).isel(time=np.arange(1,accum+1)).sum(dim='time').data

    datout = xr.Dataset(
        {
            'pr': (['rank', 'lat', 'lon'], np_datout)
        },
        coords={
            'rank': np.arange(1, eventlen+1), 'lat': datin['lat'], 'lon': datin['lon'],
        },
    )
    datout['pr'].attrs['units'] = 'mm'
    datout['pr'].attrs['long_name'] = 'Precipitation'
    datout.to_netcdf('/glade/derecho/scratch/nlybarger/data/GARD-LENS_' + domainname + '_' + experiment + '_' + accumstr + '.nc')
    # return datout


def standardize_WRF(datinpr, domainname, experiment, accum):
    """
    Standardize precipitation data to a DataArray with dimensions (event, lat, lon)
    by summing the precipitation over the time dimension
    """
    ranklist = list(datinpr['domain'].keys())
    prdatalist = [datinpr['domain'][rank] for rank in ranklist]
    # sndatalist = [datinsn['domain'][rank] for rank in ranklist]
    prdata_array = np.array(prdatalist)
    # sndata_array = np.array(sndatalist)
    datout = xr.Dataset(
        {
            # 'pr': (['event', 'y', 'x'], prdata_array+sndata_array),
            'pr': (['event', 'y', 'x'], prdata_array),
            'lat': (['y', 'x'], datinpr['domain']['001']['XLAT'].data),
            'lon': (['y', 'x'], datinpr['domain']['001']['XLONG'].data),
        },
        coords={
            'event': ranklist,
            'y': np.arange(datinpr['domain']['001']['XLAT'].shape[0]),
            'x': np.arange(datinpr['domain']['001']['XLONG'].shape[1]),
        }
        )

    datout['pr'].attrs['units'] = 'mm'
    datout['pr'].attrs['long_name'] = 'Precipitation'
    datout.to_netcdf(f'/glade/derecho/scratch/nlybarger/data/WRFpr_{domainname}_{experiment}_{accum}.accum.nc')
    # return datout