import argparse
from media_center import MediaCenter, CacheStates
from my_cluster import copy_file_as_cluster
import pprint
import os
import utils

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI-Album, an LLM-based AI auto media grouper')

    # Add arguments
    parser.add_argument('folder_path', type=str,
                        help='Path to the folder containing images')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Batch size for processing images (default: 16)')
    parser.add_argument('-sp', '--show-progress', action='store_true',
                        help='Show progress bar during processing')

    parser.add_argument('-dr', '--disable-rotation', action='store_true',
                        help='Check for media orientation and rotate them to standard direction.')
    parser.add_argument('-ded', '--disable-explicity-detection', action='store_true',
                        help='Check for exiplicit content and mark them in folder names.')

    parser.add_argument('-dl', '--distance-levels', nargs='*',
                        type=float, default=[2, 0.5],
                        help='List of distance levels for hierarchical clustering (default: [2, 0.5])')

    parser.add_argument('-o', '--output-path', type=validate_output_folder, default='',
                        help="Output path to copy files as clusters (default: ''). "
                        "Could be a path or '' (infered as <input_dir>_clustered)")
    parser.add_argument('-ot', '--output-type', nargs='+',
                        choices=['thumbnail', 'original', 'link', 'print'],
                        default=['thumbnail', 'link'],
                        help='Output types can be (one/multiple of)thumbnail, original, or link')

    parser.add_argument("-cf", "--cache-flags", type=validate_cache_arg, default = '11111',
                        help='''
Control cache settings using a binary string. Each digit represents a cache (A, B, C, D, E) in order.

Pipeline structure:
A -> B -> C -> D
         \
          -> E

where: - A: raw media material cache
 - B: media rotation detection cache
 - C: thumbnail cache
 - D: caption cache
 - E: nude detection tag cache

'1' turns a cache on, and '0' turns it off.
Turning off a cache also turns off all subsequent caches in the pipeline (A off -> B, C, D off).
However, E is independent; turning off C does not affect E.

Examples:
- '00000' turns all cache off
- '01000' works as '11000' becuase cache of B mask the unexistance of A
- '10101' turns caches A, C, and E on. B and D are off.
- '01010' actually works as '11010' due to pipeline dependencies.
- '11101' remains as it is since D and E are independent.  ''')


    args = parser.parse_args()
    if args.output_path == '':
        args.output_path = to_default_output_path(args.folder_path)

    cache_flags_str = args.cache_flags
    args.cache_flags = CacheStates(
        raw=cache_flags_str[0] == '1',
        rotate=cache_flags_str[1] == '1',
        thumb=cache_flags_str[2] == '1',
        caption=cache_flags_str[3] == '1',
        nude=cache_flags_str[4] == '1'
    )

    return args


def validate_cache_arg(cache_flags_str):
    if len(cache_flags_str) != 5 or not all(char in '01' for char in cache_flags_str):
        raise argparse.ArgumentTypeError(f"Cache flag must be a binary string of length 5, e.g., '10101'. got: {cache_flags_str}")
    return cache_flags_str


@utils.ensure_unique_path
def to_default_output_path(in_path):
    return in_path.rstrip('/').rstrip('\\') + '_clustered'

def validate_output_folder(path):
    if path == '':
        return ''
    # Check if it's a simple directory name or a valid path structure
    if os.path.basename(path) and (not os.path.exists(path) or os.path.isdir(path)):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid potential directory path "
                                         "or'print' or '' or 'default'")


def main():
    args = parse_arguments()

    # Usage
    s = MediaCenter(args.folder_path,
                    batch_size = args.batch_size,
                    check_rotation = not args.disable_rotation,
                    check_nude = not args.disable_explicity_detection,
                    show_progress_bar = args.show_progress,
                    cache_flags = args.cache_flags,
                    )

    # Compute all captions
    s.compute_all_captions()

    # Compute all tags
    s.compute_all_tags()

    if not args.output_path:
        return

    # Clustering images with specified distance levels
    clusters = s.cluster(*args.distance_levels)

    if 'print' in args.output_type:
        pprint.pprint(clusters)

    if 'original' in args.output_type:
        copy_file_as_cluster(clusters,
                             args.output_path,
                             operator = s.copy_with_meta_rotate)
    if 'thumbnail' in args.output_type:
        thumb_clusters = s.cluster_to_thumbnail(clusters)
        copy_file_as_cluster(thumb_clusters, args.output_path)
    if 'link' in args.output_type:
        copy_file_as_cluster(clusters,
                             args.output_path,
                             operator = utils.create_relative_symlink)


if __name__ == "__main__":
    main()
