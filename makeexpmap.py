from scipy import stats
from astropy import wcs
from scipy.ndimage import maximum_filter

def vigfunc(vigtab, energy=6.0):
    """A function to find the vignetting function for a given detector at a given energy


    Unit test: Test output of this function and compare to what is in the CALDB
    """

    
    from scipy.interpolate import interp1d
    from scipy.interpolate import griddata as gd
    '''Returns a vignetting function at the given energy if vig==True.
    If vig is set to False, the vignetting function will return unity regardless of the off axis angles.
    '''
    vig_theta = vigtab['THETA'][0]
    if len(np.shape(vigtab['VIGNET'][0])) == 2:
        vig_vig = vigtab['VIGNET'][0]
    elif len(np.shape(vigtab['VIGNET'][0])) == 3:
        vig_vig = vigtab['VIGNET'][0][0]
    vig_elo = vigtab['ENERG_LO'][0]
    vig_ehi = vigtab['ENERG_HI'][0]
    vig_emed = (vig_elo + vig_ehi) / 2.
    if energy is None:
        energy = vig_emed[0]
    # interpolate the vignetting function at the input energy
    grid_e, grid_th = np.meshgrid(np.array([energy]), vig_theta)
    points = np.transpose(
        [np.tile(vig_emed, len(vig_theta)), \
         np.repeat(vig_theta, len(vig_emed))]
    )
    values = vig_vig.flatten()
    int_vig = gd(points, values, (grid_e, grid_th), method='nearest').flatten()
    # define a vignetting function which takes a off-axis angle in arcmin
    # and returns a vignetting fraction
    return interp1d(vig_theta * 60., int_vig, fill_value="extrapolate")

print('deciding the right effective energy to calculate the exposure map')
arf = fits.open('HEXP_LET_v04.arf')
enlo = arf[1].data['ENERG_LO']
enhi = arf[1].data['ENERG_HI']
enmid = 0.5*(enlo+enhi)
en0arr = [0.5, 2., 8.]
en1arr = [2., 10., 24.]
for i in range(3):
    en0 = en0arr[i]
    en1 = en1arr[i]
    this_specresp = arf[1].data['SPECRESP'][np.where((enmid>= en0) & (enmid <= en1))[0]]
    this_enarr = enmid[(enmid>= en0) & (enmid <= en1)]
    # use a weighted average
    specresp_weigh = np.sum(this_specresp)/np.sum((enmid>= en0) & (enmid <= en1))
    print(en0, en1,this_enarr[np.argmin(np.abs(this_specresp - specresp_weigh))])

'''XML detector definition:
<dimensions xwidth="582" ywidth="582"/>
<wcs xrpix="291.5" yrpix="291.5" xrval="0.0" yrval="0.0" xdelt="130.e-6" ydelt="130.e-6"/>
<depfet integration="0.8e-6" clear="0.8e-6" settling="3.7e-6" type="normal"/>
poitning RA/DEC is at rawx/rawy of 291.5, 291.5, pixel size is 130e-6
# RAWY is in negative RA direction
# RAWX is in DEC direction
# since there's no rotation of the detector, 
# so the center of each rawx pixel is :
# dec = dec_cent + (rawx - 291.5)
# ra = ra_cent - (rawy - 291.5)
'''


# an example:
pts = pd.read_csv('HEXP_pointings_centered_ra00_dec00_no_rotv2.dat', delimiter=' ')
print(pts.iloc[0])
evt = fits.open('reg_11.evt')
img = fits.open('reg_11.img')

# calculate the angular distance from each detector pixel to the optical axis
rxall = np.repeat(np.arange(582), 582)
ryall = np.tile(np.arange(582), 582)
idxarr = np.indices((582, 582))
xarr = idxarr[0] - 290.5
yarr = idxarr[1] - 290.5
distarcmin = np.sqrt(xarr ** 2 + yarr ** 2) *1.34/60

# soft: - 0.95, hard - 2.9 ultra-hard - 14.895
# get the corresponding vignetting functions:
lvig = fits.open('HEXP_LET_vign_sixte_v03.fits')[1].data
# soft: 
vf = vigfunc(lvig, energy=0.95)
soft_vig = vf(distarcmin)
# hard: 
vf = vigfunc(lvig, energy=2.9)
hard_vig = vf(distarcmin)
# uhd: 
vf = vigfunc(lvig, energy=14.895)
uhd_vig = vf(distarcmin)

# coordinate transformation, and apply vignetting

imgwcs = wcs.WCS(img[0].header)
softexpmap = np.zeros_like(img[0].data)
hardexpmap = np.zeros_like(img[0].data)
uhdexpmap = np.zeros_like(img[0].data)

for i in range(len(pts)):
    racent = pts.loc[i, 'RA_cent']
    deccent = pts.loc[i, 'DEC_cent']
    expra = racent -  (ryall - 290.5)*1.34/3600
    expdec = deccent + (rxall - 290.5)*1.34/3600
    xy = imgwcs.world_to_array_index_values(expra, expdec)
    this_blankexp = np.zeros_like(img[0].data)
    this_blankexp[xy[1], xy[0]] = 25000*soft_vig.flatten()
    softexpmap = softexpmap + this_blankexp
    this_blankexp = np.zeros_like(img[0].data)
    this_blankexp[xy[1], xy[0]] = 25000*hard_vig.flatten()
    hardexpmap = hardexpmap +this_blankexp
    this_blankexp = np.zeros_like(img[0].data)
    this_blankexp[xy[1], xy[0]] = 25000*uhd_vig.flatten()    
    uhdexpmap = uhdexpmap + this_blankexp
    
# adding a maximum_filter, just in case we have some empty pixels due to coord. transformation
softexpmap = maximum_filter(softexpmap, size=2)
hardexpmap = maximum_filter(hardexpmap, size=2)
uhdexpmap = maximum_filter(uhdexpmap, size=2)

# being lazy and reuse the image headers...
soft = fits.PrimaryHDU(data=softexpmap, header=img[0].header)
soft.writeto('hexp_let_sim01_soft.exp',overwrite=True)
hard = fits.PrimaryHDU(data=hardexpmap, header=img[0].header)
hard.writeto('hexp_let_sim01_hard.exp',overwrite=True)
uhd = fits.PrimaryHDU(data=uhdexpmap, header=img[0].header)
uhd.writeto('hexp_let_sim01_uhd.exp',overwrite=True)

