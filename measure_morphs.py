import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import astropy.units as u
import statmorph
import pandas as pd

import utils.data as datutils
import utils.image as imutils
import utils.file as futils


def load_existing_results(filename):
    df = pd.read_csv(filename)
    finished_idx = df['ID'].values
    return finished_idx


def process_one(file, annulus, r1, r2, map_dir, finished_idx):
    idx = futils.find_id(file)
    if idx in finished_idx:
        print(f'Already processed {idx}')
        return idx, None
    mass_map = futils.load_map(file, map_dir)
    if mass_map is None:
        print(f'Skipping region {idx}: no map')
        return idx, None

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


def morph(map_dir, annulus=False, r1=1, r2=50):
    filename = "results/gadgetx3k_100_rin50.0kpc_rout1.0Mpc_.csv"
    finished_idx = load_existing_results(filename)
    morphs_list = []
    files_list = sorted(os.listdir(map_dir))
    total = len(files_list)
    processed = len(morphs_list)
    r1 = r1 * u.Mpc
    r2 = r2 * u.kpc

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_one, file, annulus, r1, r2, map_dir, finished_idx): file
            for file in files_list
        }

        for future in as_completed(futures):
            file = futures[future]
            idx, morph_result = future.result()
            processed += 1

            if morph_result is not None:
                morphs_list.append((idx, morph_result))

            print(f"Done {file} ({processed}/{total})")

            if len(morphs_list) % 20 == 0:
                save_results(morphs_list, r1, r2, annulus)

    save_results(morphs_list, r1, r2, annulus, final=True)


def save_results(morphs_list, r1, r2, annulus, final=False):
    if annulus:
        rad2 = r2.to(u.kpc).value
        rad1 = r1.to(u.Mpc).value
        name = f'results/gadgetx3k_{len(morphs_list)}_rin{rad2}kpc_rout{rad1}Mpc.csv'
    else:
        rad1 = r1.to(u.Mpc).value
        suffix = "final" if final else ""
        name = f'results/gadgetx3k_{len(morphs_list)}_{rad1}Mpc.csv'

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
    args = parser.parse_args()

    morph(
        map_dir=args.map_dir,
        annulus=args.annulus,
        r1=args.r1,
        r2=args.r2
    )
