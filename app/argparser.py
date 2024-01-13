from media_center import CacheStates
import textwrap
import os, glob
import pprint
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
                                    AI-Album, an LLM-based AI auto media grouper.
                                    This tool allows you to group media files in specified folders using AI algorithms.'''),
                                    formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument('folder_paths', type=str, nargs='+',
                        help=textwrap.dedent('''\
                            Provide one or more paths to folders containing images.
                            Each path should be separated by a space.
                            Example: path/to/folder1 path/to/folder2'''))

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
    parser.add_argument('-ot', '--output-type', nargs='*',
                        choices=['thumbnail', 'original', 'link', 'print'],
                        default=['thumbnail', 'link'],
                        help='Output types can be (one/multiple of)thumbnail, original, or link')

    parser.add_argument('--debug', action='store_true',
                        help='Enable function tracking timer')

    parser.add_argument("-cf", "--cache-flags", type=validate_cache_arg, default = '11111',
                        help=textwrap.dedent('''\
                            Control cache settings using a binary string. Each digit represents a cache (A, B, C, D, E) in order.

                            Pipeline structure:
                            A -> B -> C -> D
                                \
                                -> E

                            where:
                            - A: raw media material cache
                            - B: media rotation detection cache
                            - C: thumbnail cache
                            - D: caption cache
                            - E: nude detection tag cache

                            '1' turns a cache on, and '0' turns it off.
                            Turning off a cache also turns off all subsequent caches in the pipeline (A off -> B, C, D off).
                            However, E is independent; turning off C does not affect E.

                            Examples:
                            - '00000' turns all cache off
                            - '01000' works as '11000' because the cache of B masks the nonexistence of A
                            - '10101' turns caches A, C, and E on. B and D are off.
                            - '01010' actually works as '11010' due to pipeline dependencies.
                            - '11101' remains as it is since D and E are independent.'''))


    args = parser.parse_args()

    # handles wildcard expansion
    args.folder_paths = [x for p in args.folder_paths for x in glob.glob(p) ]

    if len(args.folder_paths) > 1:
        if args.output_path:
            print('Multiple input folders are provided. -o/--output-path is ignored. output folders will be infered.')
        args.output_path = ''

    cs = args.cache_flags
    args.cache_flags = CacheStates(
        raw = cs[0] == '1',
        rotate = cs[1] == '1',
        thumb = cs[2] == '1',
        caption = cs[3] == '1',
        nude = cs[4] == '1'
    )

    # Print a summary of the inputs using textwrap for better formatting
    print(textwrap.dedent("""
        User Input Summary:
        Folder Paths:
        """))
    pprint.pprint(args.folder_paths)  # Pretty print for folder paths
    print(textwrap.dedent(f"""
        Batch Size: {args.batch_size}
        Show Progress: {'Yes' if args.show_progress else 'No'}
        Disable Rotation: {'Yes' if args.disable_rotation else 'No'}
        Disable Explicit Detection: {'Yes' if args.disable_explicity_detection else 'No'}
        Distance Levels: {args.distance_levels}
        Output Path: {'Default' if args.output_path == '' else args.output_path}
        Output Types: {args.output_type}
        Cache Flags: {args.cache_flags}
        Debug Mode: {'Yes' if args.debug else 'No'}
        """))


    return args

def validate_cache_arg(cache_flags_str):
    if len(cache_flags_str) != 5 or not all(char in '01' for char in cache_flags_str):
        raise argparse.ArgumentTypeError(f"Cache flag must be a binary string of length 5, e.g., '10101'. got: {cache_flags_str}")
    return cache_flags_str

def validate_output_folder(path):
    if path == '':
        return ''
    # Check if it's a simple directory name or a valid path structure
    if os.path.basename(path) and (not os.path.exists(path) or os.path.isdir(path)):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid potential directory path "
                                         "or'print' or '' or 'default'")
