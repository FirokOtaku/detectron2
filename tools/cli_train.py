# created by Firok

from argparse import Namespace
from tools.train_net import main, launch

if __name__ == "__main__":
    from custom.cli_util import parser_addition, args_addition
    from detectron2.engine.defaults import default_argument_parser
    parser = default_argument_parser()
    parser_addition(parser)
    args: Namespace = parser.parse_args()
    args_addition(args)

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
