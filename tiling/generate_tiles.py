from datetime import datetime

from preprocessing.processing import split_positive_slide, split_negative_slide, get_otsu_threshold
from preprocessing.util import build_filename, store_slides_hdfs


def generate_positive_tiles(mgr, level, tile_size, poi_tumor, percent_overlap, max_tiles_per_slide,
                            logger, hdfs_dir, early_stopping):
    """Generate hdfs files of tiles built on tumor slides.

    Parameters
    ----------
    mgr : SlideManager object
        slide manager to access all sildes stored.
    level : int
        magnification layer number of the slide.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi_tumor : float
        minimum percentage of tissue threshold needed to save a tile
    percent_overlap: float
        percentage of overlap when generating tiles. 0.5 means we next tile will overlap with 50%
        of the tile we just generated
    max_tiles_per_slide: int
        maximum number of tiles to create from a single slide
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset

    """
    num_slides = len(mgr.annotated_slides)
    tiles_pos = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide = mgr.annotated_slides[i]

        logger.info("Working on {}".format(slide.name))
        # try:
        # create a new and unconsumed tile iterator
        tile_iter = split_positive_slide(slide, level=level,
                                         tile_size=tile_size, overlap=overlap,
                                         poi_threshold=poi_tumor)

        tiles_batch, masks_batch = list(), list()
        for tile, mask, bounds in tile_iter:
            if len(tiles_batch) % 50 == 0:
                logger.info('positive slide: {}  - tiles so far: {}'.format(i,
                                                                            len(tiles_batch)))
            if len(tiles_batch) > max_tiles_per_slide:
                break
            tiles_batch.append(tile)
            # mask = mask.reshape((tile_size, tile_size, 1)).astype(int)
            masks_batch.append(mask)

        filename = build_filename(slide.name, tile_size, poi_tumor, level, hdfs_dir)
        filename_masks = build_filename("{}{}".format(slide.name, "_masks"),
                                        tile_size,
                                        poi_tumor,
                                        level,
                                        hdfs_dir)
        num_tiles_batch = len(tiles_batch)

        store_slides_hdfs(filename, slide.name, num_tiles_batch, tiles_batch, tile_size)
        store_slides_hdfs(filename_masks, slide.name, num_tiles_batch, masks_batch, tile_size, mask=True)

        tiles_pos += len(tiles_batch)
        logger.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                      len(tiles_batch)))
        logger.info('positive tiles total: {}'.format(tiles_pos))

        # exit if reaching number of tiles generated aimed for
        if early_stopping > 0:
            if tiles_pos > early_stopping:
                break

        # except Exception as e:
        #     logger.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def generate_negative_tiles(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
                            use_upstream_filters, logger, hdfs_dir, early_stopping=0):
    """Generate hdfs files of tiles built on tumor slides.

    Parameters
    ----------
    mgr : SlideManager object
        slide manager to access all sildes stored.
    level : int
        magnification layer number of the slide.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi : float
        minimum percentage of tissue threshold needed to save a tile
    percent_overlap: float
        percentage of overlap when generating tiles. 0.5 means we next tile will overlap with 50%
        of the tile we just generated
    max_tiles_per_slide: int
        maximum number of tiles to create from a single slide
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset

    """

    num_slides = len(mgr.negative_slides)
    tiles_neg = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide = mgr.negative_slides[i]
        logger.info("Working on {}".format(slide.name))
        try:

            threshold = get_otsu_threshold(slide, level)

            # create a new and unconsumed tile iterator
            # because we have so many  negative slides we do not use overlap
            tile_iter = split_negative_slide(slide, level=level,
                                             otsu_threshold=threshold,
                                             tile_size=tile_size, overlap=overlap,
                                             poi_threshold=poi,
                                             use_upstream_filters=use_upstream_filters)

            tiles_batch = list()
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 50 == 0:
                    logger.info('negative slide: {}  - tiles so far: {}'.format(i,
                                                                                len(tiles_batch)))
                if len(tiles_batch) > max_tiles_per_slide:
                    break
                tiles_batch.append(tile)

            filename = build_filename(slide.name, tile_size, poi, level, hdfs_dir)
            num_tiles_batch = len(tiles_batch)

            store_slides_hdfs(filename, slide.name, num_tiles_batch, tiles_batch, tile_size)
            tiles_neg += len(tiles_batch)
            logger.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                          len(tiles_batch)))
            logger.info('negative tiles total: {}'.format(tiles_neg))

            # exit if reaching number of tiles generated aimed for
            if early_stopping > 0:
                if tiles_neg > early_stopping:
                    break

        except Exception as e:
            logger.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def generate_test_tiles(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
                        logger, hdfs_dir, early_stopping):
    """Generate hdfs files of tiles built on tumor slides.

    Parameters
    ----------
    mgr : SlideManager object
        slide manager to access all sildes stored.
    level : int
        magnification layer number of the slide.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi_tumor : float
        minimum percentage of tissue threshold needed to save a tile
    percent_overlap: float
        percentage of overlap when generating tiles. 0.5 means we next tile will overlap with 50%
        of the tile we just generated
    max_tiles_per_slide: int
        maximum number of tiles to create from a single slide
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset

    """
    / hack for now
    poi_tumor = poi
    print('in function')
    num_slides = len(mgr.test_slides)
    tiles_pos, tiles_neg = 0, 0
    overlap = int(tile_size * percent_overlap)
    print('Num slides: ', num_slides)
    for i in range(num_slides):
        print('i: ', i)
        slide = mgr.test_slides[i]
        bool_positive_slide = True if slide.annotations else False
        logger.info("Working on {} -  Tumor slide = {}".format(slide.name, bool_positive_slide))
        if bool_positive_slide:

            try:
                # create a new and unconsumed tile iterator
                tile_iter = split_positive_slide(slide, level=level,
                                                 tile_size=tile_size, overlap=overlap,
                                                 poi_threshold=poi_tumor)

                tiles_batch = list()
                for tile, bounds in tile_iter:
                    if len(tiles_batch) % 50 == 0:
                        logger.info('positive slide: {}  - '
                                    'tiles so far: {}'.format(i, len(tiles_batch)))
                    if len(tiles_batch) > max_tiles_per_slide:
                        break
                    tiles_batch.append(tile)

                filename = build_filename(slide.name.replace('test', 'tumor'), tile_size, poi_tumor, level, hdfs_dir)
                num_tiles_batch = len(tiles_batch)

                store_slides_hdfs(filename, slide.name.replace('test', 'tumor'), num_tiles_batch, tiles_batch, tile_size)
                tiles_pos += len(tiles_batch)
                logger.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                              len(tiles_batch)))
                logger.info('positive tiles total: {}'.format(tiles_pos))

                if early_stopping > 0:
                    if tiles_pos > early_stopping and tiles_neg > early_stopping:
                        break

            except Exception as e:
                logger.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))

        else:
            try:

                threshold = get_otsu_threshold(slide, level)

                # create a new and unconsumed tile iterator
                # because we have so many  negative slides we do not use overlap
                tile_iter = split_negative_slide(slide, level=level,
                                                 otsu_threshold=threshold,
                                                 tile_size=tile_size, overlap=overlap,
                                                 poi_threshold=poi,
                                                 use_upstream_filters=False)

                tiles_batch = list()
                for tile, bounds in tile_iter:
                    if len(tiles_batch) % 50 == 0:
                        logger.info('negative slide: {}  - tiles so far: {}'.format(i,
                                                                                    len(tiles_batch)))
                    if len(tiles_batch) > max_tiles_per_slide:
                        break
                    tiles_batch.append(tile)

                filename = build_filename(slide.name.replace('test', 'normal'), tile_size, poi, level, hdfs_dir)
                num_tiles_batch = len(tiles_batch)

                store_slides_hdfs(filename, slide.name.replace('test', 'normal'), num_tiles_batch, tiles_batch, tile_size)
                tiles_neg += len(tiles_batch)
                logger.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                              len(tiles_batch)))
                logger.info('negative tiles total: {}'.format(tiles_neg))

                if early_stopping > 0:
                    if tiles_pos > early_stopping and tiles_neg > early_stopping:
                        break

            except Exception as e:
                logger.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))