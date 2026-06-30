import utils.file as futils
import utils.data as datutils
import argparse

def main(mah_dir):
	mah_dict = futils.get_mah_all(mah_dir)
	ma, _, aexp_bins = datutils.interp_ma(mah_dict)
	am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                  min_mass_bin=0.01, log_spacing=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', type=str, 
		default='/Users/kabelo/clusters/data/gadgetx3k/AHFHaloHistory',
		 help='Directory containing AHF outputs')
	args = parser.parse_args()
	main(args.dir)
