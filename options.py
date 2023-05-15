import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PDADA')
        parser.add_argument('--desc', type=str, default="default", help='description')
        parser.add_argument('--session', type=str, default="default")

        ''' Dataset '''
        parser.add_argument('--dataset', type=str, default='office_home',
                            choices=['office', 'office_home', 'visda17', 'minidomainnet'],
                            help='the name of dataset')
        parser.add_argument('--source', type=str, default='Art', help='source domain')
        parser.add_argument('--target', type=str, default='Clipart', help='target domain')
        parser.add_argument('--budget', type=float, default=0.05, help='budget for active learning')

        ''' running mode '''
        parser.add_argument('--resume', type=str, default=None, help='path to pth')
        parser.add_argument('--resume_training', action='store_true', default=False)
        parser.add_argument('--testonly', action='store_true', default=False)

        ''' model '''
        parser.add_argument('--method', type=str, default='PDADA', help='method')
        parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50'], help='network to use')
        parser.add_argument('--scratch', default=False, action='store_true')

        ''' optimization '''
        parser.add_argument('--bs', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
        parser.add_argument('--max_epoch', type=int, default=100)
        parser.add_argument('--start_step', type=int, default=-1)

        ''' resource options '''
        parser.add_argument('--num_workers', type=int, default=9)
        parser.add_argument('--unl_num_workers', type=int, default=9)

        ''' logging '''
        parser.add_argument('--dontlog', action='store_true', default=True, help='Not logging')
        parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                            help='batches to wait before logging '
                                 'training status')
        parser.add_argument('--save_interval', type=int, default=100, metavar='N',
                            help='batches to wait before saving a model')

        ''' misc '''
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
        parser.add_argument('--lam', type=float, default=0.1)
        self.parser = parser

    def modify_command_options(self, args):
        if (args.dataset == 'office_home'):
            args.ncls = 65
            args.k = 20
        elif (args.dataset == 'visda17'):
            args.ncls = 12
            args.k = 5
        elif (args.dataset == 'office'):
            args.ncls = 31
            args.k = 9
        elif (args.dataset == 'minidomainnet'):
            args.ncls = 126
            args.k = 5
        else:
            raise NotImplementedError

        ''' Modify session '''
        if args.dataset == 'office_home':
            data_code = 'O'
        elif args.dataset == 'visda17':
            data_code = 'V'
        elif args.dataset == 'office':
            data_code = 'O31'
        elif args.dataset == 'minidomainnet':
            data_code = 'D'
        else:
            raise NotImplementedError
        args.session = args.session + '-b{}_{}{}{}2{}'.format(
            args.budget,
            data_code,
            args.net.capitalize()[0],
            args.source.capitalize()[0],
            args.target.capitalize()[0]
        )

        ''' Logging '''
        args.wandb_log = not args.dontlog

        return args

    def parse(self):
        args = self.parser.parse_args()
        return args
