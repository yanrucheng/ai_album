from media_center import MediaCenter
from my_cluster import operate_file_as_cluster
import utils
from function_tracker import global_tracker
from argparser import parse_arguments, to_default_output_path
from log_config import set_logger_config

import pprint
import logging
logger = logging.getLogger(__name__)

def app(in_folder, args):
    # Usage
    s = MediaCenter(in_folder,
                    batch_size = args.batch_size,
                    check_rotation = not args.disable_rotation,
                    show_progress_bar = not args.disable_progress,
                    cache_flags = args.cache_flags,
                    datum = args.map_datum,
                    )
    s.compute_all_cache()

    thumb_cluster = s.thumbnail_cluster(*args.distance_levels)
    cluster = s.full_cluster(*args.distance_levels)

    output_path = args.output_path
    if output_path == '':
        output_path = to_default_output_path(in_folder)

    if 'print' in args.output_type:
        pprint.pprint(cluster)

    if 'original' in args.output_type:
        operate_file_as_cluster(cluster, output_path,
                                operator = utils.safe_move)

    if 'thumbnail' in args.output_type:
        operate_file_as_cluster(thumb_cluster, output_path,
                                operator = utils.copy_with_meta)

    if 'link' in args.output_type:
        operate_file_as_cluster(cluster, output_path,
                                operator = utils.create_relative_symlink)


def main():
    args = parse_arguments()

    if args.debug:
        global_tracker.enable()
        set_logger_config()

    for f in args.folder_paths:
        app(f, args)

    if args.debug:
        global_tracker.report()


if __name__ == "__main__":
    main()
