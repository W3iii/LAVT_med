import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--data_root', default='../dataset_2classes',
                        help='root of lung-nodule slice dataset (images/, masks/, annotations/)')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=None, type=int,
                        help='deprecated square input size; use --img_h/--img_w for rectangular CT input')
    parser.add_argument('--img_h', default=384, type=int,
                        help='input image height after pad/crop')
    parser.add_argument('--img_w', default=512, type=int,
                        help='input image width after pad/crop')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='local rank for DistributedDataParallel; -1 = single-GPU mode')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt_one', help='name to identify the model')
    parser.add_argument('--neg_ratio', default=0.1, type=float,
                        help='negative-to-positive sampling ratio per epoch')
    parser.add_argument('--n_soft_tokens', default=4, type=int,
                        help='number of learnable soft prompt tokens')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pred_dir', default='./pred_results',
                        help='where test.py writes prediction masks when --save_pred is set')
    parser.add_argument('--save_pred', action='store_true',
                        help='if set, test.py saves per-sample predicted masks as PNG')
    parser.add_argument('--cc_stats_json', default='',
                        help='if set, test.py writes per-slice CC sizes + TP/FP labels '
                             'to this JSON path for downstream FP-size analysis')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='random seed for negative resampling')
    parser.add_argument('--split', default='val', help='only used when testing')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
