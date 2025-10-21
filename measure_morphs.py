import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import astropy.units as u
import statmorph
import pandas as pd

import utils.data as datutils
import utils.image as imutils
import utils.file as futils
import warnings
from pandas.errors import ParserWarning

warnings.simplefilter(action='ignore', category=ParserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_existing_results(filename):
    df = pd.read_csv(filename)
    finished_idx = df['ID'].values
    return finished_idx


def process_one(file, annulus, r1, r2, map_dir, use_vir=False):
    idx = futils.find_id(file)

    mass_map = futils.load_map(file, map_dir)
    if mass_map is None:
        print(f'Skipping region {idx}: no map')
        return idx, None

    if use_vir:
        r1, _ = futils.get_virial_radius(idx)

    pixr1 = datutils.real2pix(r1, mass_map)
    pixr2 = datutils.real2pix(r2, mass_map)
    center = (int(len(mass_map[1]) // 2), int(len(mass_map[0]) // 2))
    print(f'Processing {idx}...')

    if annulus:
        segmap = imutils.annular_mask(mass_map, center, pixr2, pixr1)
    else:
        segmap = imutils.circular_segmap(mass_map, center, pixr1)

    morph = statmorph.source_morphology(
        mass_map,
        segmap,
        gain=2.25,
    )
    return idx, morph[0]


def morph(map_dir, annulus=False, r1=1, r2=50, out_dir='.', use_vir=False):
    morphs_list = []
    files_list = sorted(os.listdir(map_dir))
    total = len(files_list)
    processed = len(morphs_list)
    r1 = r1 * u.Mpc
    r2 = r2 * u.kpc

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_one, file, annulus, r1, r2, map_dir, use_vir): file
            for file in files_list
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                idx, morph_result = future.result()
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            processed += 1

            if morph_result is not None:
                morphs_list.append((idx, morph_result))

            print(f"Done {file} ({processed}/{total})")

            if len(morphs_list) % 20 == 0:
                save_results(morphs_list, r1, r2, annulus,
                             out_dir, use_vir=use_vir)

    save_results(morphs_list, r1, r2, annulus, out_dir, use_vir=use_vir)


def save_results(morphs_list, r1, r2, annulus, out_dir, use_vir=False):
    if annulus:
        rad2 = r2.to(u.kpc).value
        rad1 = r1.to(u.Mpc).value
        if use_vir:
            name = f'results/{out_dir}/rin{rad2}kpc_rout_R200c_{len(morphs_list)}.csv'
        else:
            name = f'results/{out_dir}/rin{rad2}kpc_rout{rad1}Mpc_{len(morphs_list)}.csv'
    else:
        if use_vir:
            name = f'results/{out_dir}/rout_R200c_{len(morphs_list)}.csv'
        rad1 = r1.to(u.Mpc).value
        name = f'results/{out_dir}/r_{rad1}Mpc_{len(morphs_list)}.csv'

    # ensure output directory exists
    outdir = os.path.dirname(name) or 'results'
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving {len(morphs_list)} morph results -> {name}")

    futils.create_morph_df(morphs_list, name=name, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir", type=str, required=True,
                        help="Directory containing maps")
    parser.add_argument("--r1", type=float, default=1.0,
                        help="Outer radius in Mpc, default=1")
    parser.add_argument("--r2", type=float, default=50.0,
                        help="Inner radius in kpc, default=50")
    parser.add_argument("--annulus", action="store_true",
                        help="Use annulus mask")
    parser.add_argument("--out_dir", type=str, default='./',
                        help="Output directory under results/")
    parser.add_argument("--vir", action="store_true",
                        help="Use virial radius for r1")
    args = parser.parse_args()

    morph(
        map_dir=args.map_dir,
        annulus=args.annulus,
        r1=args.r1,
        r2=args.r2,
        out_dir=args.out_dir,
        use_vir=args.vir
    )
