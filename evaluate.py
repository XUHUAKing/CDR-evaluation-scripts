"""
API for evaluating CDR dataset: PNCC, PSNR, SSIM
Lastest update: Jun. 25, 2021
"""
import os
import csv
import re
import shutil
import argparse
import io


class CDREvaluator:
    def __init__(self, csvpath, gtpath, predpath, outpath):
        if os.path.exists('./VGG_Model/imagenet-vgg-verydeep-19.mat'):
            print("please download the VGG checkpoint following README.md first")
            exit()
        self.csvpath = csvpath
        self.gtpath = gtpath
        self.predpath = predpath
        self.outpath = outpath # output folder storing txt
        self.pncc_path = None
        self.psnr_path = None
        self.ssim_path = None

    def evaluate_pncc(self):
        # command to evaluate pncc metric
        self.pncc_path = os.path.join(self.outpath, "pncc.txt")
        cmd = f"python pncc_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.pncc_path}"
        print("running: %s"%(cmd))
        # os.system(cmd)

    def evaluate_psnr(self):
        # command to evaluate psnr metric
        self.psnr_path = os.path.join(self.outpath, "psnr.txt")
        cmd = f"python psnr_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.psnr_path}"
        print("running: %s"%(cmd))
        # os.system(cmd)

    def evaluate_ssim(self):
        # command to evaluate ssim metric
        self.ssim_path = os.path.join(self.outpath, "ssim.txt")
        cmd = f"python ssim_2dirs.py -d0 {self.gtpath} -d1 {self.predpath} -o {self.ssim_path}"
        print("running: %s"%(cmd))
        # os.system(cmd)

    def combine_results(self):
        # ensure all metrics are ready to combine
        assert (self.pncc_path is not None) and (self.psnr_path is not None) and (self.ssim_path is not None)
        # parse txt metrics files
        pncc_file = open(self.pncc_path, 'r')
        pncc_metrics = self._parse_txt(pncc_file)
        psnr_file = open(self.psnr_path, 'r')
        psnr_metrics = self._parse_txt(psnr_file)
        ssim_file = open(self.ssim_path, 'r')
        ssim_metrics = self._parse_txt(ssim_file)
        assert len(pncc_metrics.keys()) == len(psnr_metrics.keys()) == len(ssim_metrics.keys())
        # parse csv files
        csv_file = io.open(self.csvpath, encoding='utf-8-sig')
        rows = csv.DictReader(csv_file)

        cnts = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0}
        pnccs = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0}
        psnrs = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0}
        ssims = {'BRBT': 0, 'BRST': 0, 'SRST': 0, 'weak': 0, 'medium': 0, 'strong': 0, 'ghost_yes': 0, 'ghost_no': 0}

        cnt = 0
        for row in rows:
            new_name = row['name'].replace('/', '_')
            if new_name in pncc_metrics:
                cnts[row['type']] += 1
                cnts[row['reflection']] += 1
                # ghosting
                ghost_name = 'ghost_yes' if row['ghost'] == '1' else 'ghost_no'
                cnts[ghost_name] += 1
                psnrs[ghost_name] += float(psnr_metrics[new_name])
                ssims[ghost_name] += float(ssim_metrics[new_name])
                pnccs[ghost_name] += float(pncc_metrics[new_name])
                # BRBT / BRST / SRST
                psnrs[row['type']] += float(psnr_metrics[new_name])
                ssims[row['type']] += float(ssim_metrics[new_name])
                pnccs[row['type']] += float(pncc_metrics[new_name])
                # weak / medium / strong
                psnrs[row['reflection']] += float(psnr_metrics[new_name])
                ssims[row['reflection']] += float(ssim_metrics[new_name])
                pnccs[row['reflection']] += float(pncc_metrics[new_name])

                cnt += 1

        print(cnt, cnts)
        for key in cnts:
            print("%10s %03d %.3f %.3f %.3f" % (
            key, cnts[key],
            psnrs[key] / float(cnts[key] + 1),
            ssims[key] / float(cnts[key] + 1),
            pnccs[key] / float(cnts[key] + 1)))

    def _parse_txt(self, file):
        txt_rows = file.readlines()
        metrics = {}
        for row in txt_rows:
            contents = row[:-1].split(' ')
            id = "_".join(contents[0].split("_")[:3])
            metrics[id] = float(contents[1])
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
    parser.add_argument('--pncc', action='store_true', help='evaluate on pncc')
    parser.add_argument('--psnr', action='store_true', help='evaluate on psnr')
    parser.add_argument('--ssim', action='store_true', help='evaluate on ssim')

    return parser

def parse_args(parser):
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    evaluator = CDREvaluator(args.csvpath, args.gtpath, args.predpath, args.output)
    # evaluate on every metric in turns
    if args.psnr:
        evaluator.evaluate_psnr()
    if args.ssim:
        evaluator.evaluate_ssim()
    if args.pncc:
        evaluator.evaluate_pncc()
    # combine evaluation metrics based on categories
    print("combing all metrics and output category-wise results...")
    evaluator.combine_results()

    print("Done! Benchmarking results are ready!")