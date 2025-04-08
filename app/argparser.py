from media_center import CacheStates
import textwrap
import os, glob
import pprint
import argparse
import utils

@utils.ensure_unique_path
def to_default_output_path(in_path):
    return in_path.rstrip('/').rstrip('\\') + '-clustered'

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

    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing images (default: 16)')
    parser.add_argument('--disable-progress', action='store_true',
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

    parser.add_argument('-lang', '--language',
                        choices=['en', 'zh'],
                        default='en',
                        help='Select the language for clustered folder names. (default: en) (zh=chinese, en=english)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable function tracking timer')

    parser.add_argument("-cf", "--cache-flags", type=validate_cache_arg, default = '1111111',
                        help=textwrap.dedent('''\
                            Control cache settings using a binary string. Each digit represents a cache (A, B, C, D, E) in order.

                            Pipeline structure:
                              -> B
                             /
                            A -> C -> D -> E -> G
                             \\
                              -> F

                            where:
                            - A: raw media material cache
                            - B: metadata
                            - C: media rotation detection cache
                            - D: thumbnail cache
                            - E: caption cache
                            - F: nude detection tag cache
                            - G: title generation cache

                            '1' turns a cache on, and '0' turns it off.
                            Turning off a cache also turns off all subsequent caches in the pipeline (A off -> B, C, D off).
                            However, E is independent; turning off C does not affect E.

                            Examples:
                            - '0000000' turns all cache off
                            - '0100000' works as '11000' because the cache of B masks the nonexistence of A
                            - '1010010' turns caches A, C, and E on. B and D are off.
                            - '0101000' actually works as '11010' due to pipeline dependencies.
                            - '1110100' remains as it is since D and E are independent.'''))


    args = parser.parse_args()

    # handles wildcard expansion
    args.folder_paths = [
        path
        for path_pattern in args.folder_paths
        for path in glob.glob(path_pattern)
        if not any(k in path for k in ('-clustered', 'tmp'))
    ]

    if len(args.folder_paths) > 1:
        if args.output_path:
            print('Multiple input folders are provided. -o/--output-path is ignored. output folders will be infered.')
        args.output_path = ''

    cs = args.cache_flags
    args.cache_flags = CacheStates(
        raw = cs[0] == '1',
        meta = cs[1] == '1',
        rotate = cs[2] == '1',
        thumb = cs[3] == '1',
        caption = cs[4] == '1',
        nude = cs[5] == '1',
        title = cs[6] == '1',
    )

    # Print a summary of the inputs using textwrap for better formatting
    print(textwrap.dedent("""
        User Input Summary:
        Folder Paths:
        """))
    pprint.pprint(args.folder_paths)  # Pretty print for folder paths
    lang_str = {'en': 'English', 'zh': 'Chinese'}[args.language]
    print(textwrap.dedent(f"""
        Batch Size: {args.batch_size}
        Show Progress Bar: {'No' if args.disable_progress else 'Yes'}
        Check Media Rotation: {'Off' if args.disable_rotation else 'On'}
        Check Explicit Detection: {'Off' if args.disable_explicity_detection else 'On'}
        Distance Levels: {args.distance_levels}
        Output Path: {'Default' if args.output_path == '' else args.output_path}
        Output Types: {args.output_type}
        Cache Flags: {args.cache_flags}
        Debug Mode: {'Yes' if args.debug else 'No'}
        Language: {lang_str}
        """))


    return args

def validate_cache_arg(cache_flags_str):
    if len(cache_flags_str) != 7 or not all(char in '01' for char in cache_flags_str):
        raise argparse.ArgumentTypeError(f"Cache flag must be a binary string of length 7, e.g., '1010100'. got: {cache_flags_str}")
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
