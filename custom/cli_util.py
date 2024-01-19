# created by Firok

def register_dataset(name: str):
    from detectron2.data.datasets import register_coco_instances
    file_annotation = "datasets/" + name + "/coco.json"
    folder_image = "datasets/" + name + "/images"

    register_coco_instances(
        name,
        {},
        file_annotation,
        folder_image
    )


def parser_addition(parser):
    parser.add_argument("--reg-dataset-train", type=str, required=False, help="register new dataset into detectron2 for training")
    parser.add_argument("--reg-dataset-test", type=str, required=False, help="register new dataset into detectron2 for testing")

def args_addition(args):
    if args.reg_dataset_train is not None:
        register_dataset(args.reg_dataset_train)
    if args.reg_dataset_test is not None:
        register_dataset(args.reg_dataset_test)
