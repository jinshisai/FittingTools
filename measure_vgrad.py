import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import astropy.io.fits as fits
import mpl_toolkits.axes_grid1
import matplotlib.patches as patches

import pyfigures as pyfg
from imfits import Imfits


### setting for figures
#mpl.use('Agg')
#mpl.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
plt.rcParams['font.size'] = 18           # fontsize
#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth



### constants
auTOkm = 1.495978707e8  # AU --> km
auTOcm = 1.495978707e13 # AU --> cm
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au
pcTOcm = 3.09e18        # pc --> cm



### functions
def func_G93(del_ra, del_dec, v0, a, b):
	# See Goodman et al. (1993)
	vlsr = v0 + a*del_ra + b*del_dec
	return vlsr


def _func_G93(xdata,*args):
	del_ra, del_dec = xdata
	#print (*args)
	ans = func_G93(del_ra, del_dec, *args)
	return ans


def measure_vgrad(fname, p0, rfit=None, dist=140.,
 outname=None, outfig=False):
	'''
	Measure a velocity gradient and its P.A. Based on the method introduced in Goodman et al. (1993).

	Parameters
	----------
	fname: fits file name
	p0: initial parameter estimate
	rfit: Region where fitting is performed. Should be given in radius.
	dist: Distance to the object.
	outname: Output name if outfig=True.
	outfig: Plot the fitting result?
	'''

	cube  = Imfits(fname)
	data  = cube.data[0,1,:,:] # moment I
	derr  = cube.data[0,2,:,:] # error of moment I
	xx    = cube.xx
	yy    = cube.yy
	naxis = cube.naxis

	xx = xx*3600. # deg --> arcsec
	yy = yy*3600.

	# delta
	dx = np.abs(xx[0,1] - xx[0,0])
	dy = np.abs(yy[1,0] - yy[0,0])
	#print (xx.shape)

	# beam
	bmaj, bmin, bpa = cube.beam # as, as, deg

	# radius
	rr = np.sqrt(xx*xx + yy*yy)

	# sampling
	step = int(bmin/dx*0.5)
	ny, nx = xx.shape
	#print (step)
	xx_fit = xx[0:ny:step, 0:nx:step]
	yy_fit = yy[0:ny:step, 0:nx:step]
	data_fit = data[0:ny:step, 0:nx:step]
	#print (data_fit.shape)


	if rfit:
		where_fit = np.where(rr <= rfit)
		data_fit = data[where_fit]
		derr_fit = derr[where_fit]
		xx_fit   = xx[where_fit]
		yy_fit   = yy[where_fit]
	else:
		data_fit = data
		derr_fit = derr
		xx_fit   = xx
		yy_fit   = yy

	#plt.scatter(xx_fit, yy_fit, c=data_fit, alpha=0.3)
	#plt.show()


	# exclude nan
	xx_fit   = xx_fit[~np.isnan(data_fit)]
	yy_fit   = yy_fit[~np.isnan(data_fit)]
	derr_fit = derr_fit[~np.isnan(data_fit)]
	data_fit = data_fit[~np.isnan(data_fit)]
	#print (xx_fit)


	# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
	xdata = np.vstack((xx_fit, yy_fit)) # or xx.ravel()
	#print (xdata)

	# fitting
	popt, pcov = curve_fit(_func_G93, xdata, data_fit, p0,
		sigma=derr_fit, absolute_sigma=True)
	perr       = np.sqrt(np.diag(pcov))
	v0, a, b   = popt
	v0_err, a_err, b_err = perr

	# velocity gradient
	vgrad    = (a*a + b*b)**0.5/dist/auTOpc # km s^-1 pc^-1
	th_vgrad = np.arctan2(a,b)              # radians

	# error of vgrad through the error propagation
	c01       = (a*a + b*b)**(-0.5)/dist/auTOpc
	vgrad_err = c01*np.sqrt((a*a_err)*(a*a_err) + (b*b_err)*(b*b_err))

	# error of th_vgrad through the error propagation
	costh2 = np.cos(th_vgrad)*np.cos(th_vgrad)
	sinth2 = np.sin(th_vgrad)*np.sin(th_vgrad)
	th_vgrad_err = np.sqrt(
		(costh2*a_err/b)*(costh2*a_err/b)
		+ (sinth2*b_err/a)*(sinth2*b_err/a))

	# unit
	th_vgrad     = th_vgrad*180./np.pi     # rad --> degree
	th_vgrad_err = th_vgrad_err*180./np.pi # rad --> degree

	# output results
	print ('(v0,a,b)=(%.2e,%.2e,%.2e)'%(popt[0],popt[1],popt[2]))
	print ('(sig_v0,sig_a,sig_b)=(%.2e,%.2e,%.2e)'%(perr[0],perr[1],perr[2]))
	print ('Vgrad: %.2f +/- %.2f km/s/pc'%(vgrad,vgrad_err))
	print ('P.A.: %.1f +/- %.1f deg'%(th_vgrad,th_vgrad_err))


	# output image for check
	if outfig:
		vlsr   = func_G93(xx,yy,*popt)
		xmin   = xx[0,0]
		xmax   = xx[-1,-1]
		ymin   = yy[0,0]
		ymax   = yy[-1,-1]
		extent = (xmin,xmax,ymin,ymax)
		#print (vlsr)

		fig = plt.figure(figsize=(11.69,8.27))
		ax  = fig.add_subplot(111)
		im = ax.imshow(vlsr, origin='lower',cmap='jet',extent=extent, vmin=6.6, vmax=7.4)

		# direction of the velocity gradient
		costh = np.cos(th_vgrad*np.pi/180.)
		sinth = np.sin(th_vgrad*np.pi/180.)
		mrot = np.array([[costh, -sinth],
			[sinth, costh]])
		p01 = np.array([0,60])
		p02 = np.array([0,-60])
		p01_rot = np.dot(p01,mrot)
		p02_rot = np.dot(p02,mrot)
		ax.plot([p01_rot[0],p02_rot[0]],[p01_rot[1],p02_rot[1]],ls='--',lw=1, c='k',sketch_params=0.5)

		# colorbar
		divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
		cax     = divider.append_axes('right','3%', pad='0%')
		cbar    = fig.colorbar(im, cax = cax)
		cbar.set_label(r'$\mathrm{km\ s^{-1}}$')

		# labels
		ax.set_xlabel('RA offset (arcsec)')
		ax.set_ylabel('DEC offset (arcsec)')


		if outname:
			pass
		else:
			outname = 'measure_vgrad_res'
		plt.savefig(outname+'.pdf',transparent=True)
		#plt.show()

		return ax, popt, perr
	else:
		return v0, v0_err, vgrad, vgrad_err, th_vgrad, th_vgrad_err



if __name__ == '__main__':
	fitsdata = ['../fitsimages/iras15398_c18o21_comb_aca-apex_v3_dv011_pbcor_moments_sig5.fits',
	'../fitsimages/l1527_c18o21_comb_aca-iram30m_v3_dv0084_pbcor_moments_sig5.fits',
	'../fitsimages/tmc-1a_c18o21_comb_aca-iram30m_v3_dv0084_pbcor_moments_sig5.fits',
	'../fitsimages/l1489_c18o21_aca_iram30m_combv5_cintr_pbcor_moments_sig5.fits']

	outname  = 'measure_vgrad_werr_res'

	obs      = ['b228', 'l1527', 'tmc-1a', 'l1489']

	p0s      = [[5.4,-0.0005, -0.001],
	[5.4,-0.001, -0.001],
	[6.4,-0.005, -0.005],
	[7.2, -0.005, -0.005]
	]

	imscale=[-60,60,-60,60]

	# loop
	for s in range(len(obs)):
		source    = obs[s]
		outname_s = outname + '_' + source
		p0        = p0s[s]
		fits_s    = fitsdata[s]


		rfit_list = np.arange(10.,70.,10.)
		nrfit     = len(rfit_list)

		with open(outname_s + '.txt', mode='w+') as f:
			f.write('r_fit v0 v0_err vgrad vgrad_err th_vgrad th_vgrad_err\n')
			f.write('#arcsec km/s km/s km/s/pc km/s/pc degree degree\n')
			f.write('#r_fit: radius range for the fit (arcsec)\n\n')

			for i in range(nrfit):
				rfit = rfit_list[i]
				v0, v0_err, vgrad, vgrad_err, th_vgrad, th_vgrad_err = measure_vgrad(fits_s, p0, rfit=rfit,
				 outname=outname, outfig=False)

				f.write('%i %.3e %.3e %.3e %.3e %.3e %.3e\n'%(rfit, v0, v0_err, vgrad, vgrad_err, th_vgrad, th_vgrad_err))