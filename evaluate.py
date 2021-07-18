"""
API for evaluating CDR dataset: PNCC, PSNR, SSIM, NCC
Lastest update: July. 18, 2021
"""
import os
import csv
import re
import shutil
import argparse
import io


class CDREvaluator:
    def __init__(self, csvpath, gtpath, predpath, outpath):
        if not os.path.exists('./VGG_Model/imagenet-vgg-verydeep-19.mat'):
            print("please download the VGG checkpoint following README.md first")
            exit()
        self.csvpath = csvpath
        self.gtpath = gtpath
        self.predpath = predpath
        self.outpath = outpath # output folder storing txt
        self.ncc_path = None
        self.pncc_path = None
        self.psnr_path = None
        self.ssim_path = None

    def evaluate_ncc(self):
        # command to evaluate ncc metric
        self.ncc_path = os.path.join(self.outpath, "ncc.txt")
        cmd = f"python ncc_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.ncc_path}"
        print("calculating NCC metric...")
        os.system(cmd)

    def evaluate_pncc(self):
        # command to evaluate pncc metric
        self.pncc_path = os.path.join(self.outpath, "pncc.txt")
        cmd = f"python pncc_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.pncc_path}"
        print("calculating PNCC metric...")
        os.system(cmd)

    def evaluate_psnr(self):
        # command to evaluate psnr metric
        self.psnr_path = os.path.join(self.outpath, "psnr.txt")
        cmd = f"python psnr_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.psnr_path}"
        print("calculating PSNR metric...")
        os.system(cmd)

    def evaluate_ssim(self):
        # command to evaluate ssim metric
        self.ssim_path = os.path.join(self.outpath, "ssim.txt")
        cmd = f"python ssim_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.ssim_path}"
        print("calculating SSIM metric...")
        os.system(cmd)

    def combine_results(self):
        # ensure all metrics are ready to combine (this constraint will be removed in the future)
        assert (self.ncc_path is not None) and (self.psnr_path is not None) and (self.ssim_path is not None)
        # parse txt metrics files
        ncc_file = open(self.ncc_path, 'r')
        ncc_metrics = self._parse_txt(ncc_file)
        psnr_file = open(self.psnr_path, 'r')
        psnr_metrics = self._parse_txt(psnr_file)
        ssim_file = open(self.ssim_path, 'r')
        ssim_metrics = self._parse_txt(ssim_file)
        if not (len(ncc_metrics.keys()) == len(psnr_metrics.keys()) == len(ssim_metrics.keys())):
           print("!!!WARNING: the number of images are different in NCC, PSNR and SSIM txts")
        # parse csv files
        csv_file = io.open(self.csvpath, encoding='utf-8-sig')
        rows = csv.DictReader(csv_file)

        cnts = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0, 'all': 0}
        nccs = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0, 'all': 0}
        psnrs = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0, 'all': 0}
        ssims = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0, 'all': 0}

        for row in rows:
            new_name = row['name'].replace('/', '_')
            if new_name in psnr_metrics:
                # count for ALL
                num_crops_for_curr_id = int(psnr_metrics[new_name][1])
                cnts['all'] += num_crops_for_curr_id
                psnrs['all'] += float(psnr_metrics[new_name][0])
                ssims['all'] += float(ssim_metrics[new_name][0])
                nccs['all'] += float(ncc_metrics[new_name][0])
                # count for each type
                cnts[row['type']] += num_crops_for_curr_id
                cnts[row['reflection']] += num_crops_for_curr_id
                # ghosting
                ghost_name = 'ghost_yes' if row['ghost'] == '1' else 'ghost_no'
                cnts[ghost_name] += num_crops_for_curr_id
                psnrs[ghost_name] += float(psnr_metrics[new_name][0])
                ssims[ghost_name] += float(ssim_metrics[new_name][0])
                nccs[ghost_name] += float(ncc_metrics[new_name][0])
                # BRBT / BRST / SRST
                psnrs[row['type']] += float(psnr_metrics[new_name][0])
                ssims[row['type']] += float(ssim_metrics[new_name][0])
                nccs[row['type']] += float(ncc_metrics[new_name][0])
                # weak / medium / strong
                psnrs[row['reflection']] += float(psnr_metrics[new_name][0])
                ssims[row['reflection']] += float(ssim_metrics[new_name][0])
                nccs[row['reflection']] += float(ncc_metrics[new_name][0])

        print(cnts)
        # txt path saving the conclusive results
        result_path = os.path.join(self.outpath, "result.txt")
        f = open(result_path,'w')
        for key in cnts:
            res = "%10s %03d %.3f %.3f %.3f" % (
            key, cnts[key],
            psnrs[key] / float(cnts[key]),
            ssims[key] / float(cnts[key]),
            nccs[key] / float(cnts[key]))
            f.writelines('%s\n' % (res))
            print(res)

    def _parse_txt(self, file):
        txt_rows = file.readlines()
        metrics = {}
        for row in txt_rows:
            contents = row[:-1].split(' ')
            id = "_".join(contents[0].split("_")[:3])
            # metrics: {id: (metric value sume, counter)}
            if id in metrics:
                # multiple crops for this images, update metric sum and counter
                metrics[id] = (metrics[id][0] + float(contents[1]), metrics[id][1] + 1)
            else:
                # initialize
                metrics[id] = (float(contents[1]), 1)
        return metrics



def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    # paths
    parser.add_argument('--csvpath', type=str, required=True,
                        help='path to csv')
    parser.add_argument('--gtpath', type=str, required=True,
                        help='path to ground truth, e.g. data/T')
    parser.add_argument('--predpath', type=str, required=True,
                        help='path to predicted outputs')
    parser.add_argument('--output', type=str, required=True,
                        help='path to store the metric results (txt, results)')
    # dataset choices
    parser.add_argument('--ncc', action='store_true', help='evaluate on ncc')
    parser.add_argument('--psnr', action='store_true', help='evaluate on psnr')
    parser.add_argument('--ssim', action='store_true', help='evaluate on ssim')

    return parser

def parse_args(parser):
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    # create output dir if not exists
    os.makedirs(args.output, exist_ok=True)

    evaluator = CDREvaluator(args.csvpath, args.gtpath, args.predpath, args.output)
    # evaluate on every metric in turns
    if args.psnr:
        evaluator.evaluate_psnr()
    if args.ssim:
        evaluator.evaluate_ssim()
    if args.ncc:
        evaluator.evaluate_ncc()
    # combine evaluation metrics based on categories
    print("combing all metrics and output category-wise results...")
    evaluator.combine_results()

    print("Done! Benchmarking results are ready!")