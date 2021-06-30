# CDR-evaluation-scripts
Evaluation codes for benchmarking

### Pre-Steps
Please test and output your predicted results before using this scripts. Your testing pipeline should be:
1. Use *_M* image as input to your model 
2. Save your output transmission using the same name as input, under a folder named "T/"
3. Save your output reflection using the same name as input (if any), under a folder named "R/"

### Setup
Need to download tensorflow and Python 3.6+, if you are using server 1, directly use this conda environment:
```
conda activate perceptual-reflection-removal
```

Please download the VGG19 models for PNCC evaluation from this [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cleiaa_connect_ust_hk/EmGaVX18EfFDn49CA_eTeEoBLhhvwv7aspvGp-GWD_sFTQ?e=GtOqnc).
Then put it under the folder ```VGG_Model/```

### Usage
To test T (transmission image) on VAL:
```
python evaluate.py --csvpath annotation.csv --gtpath ./val/T --predpath ./pred/T --output ./results --pncc --psnr --ssim
```


To test R (reflection image) on VAL:
```
python evaluate.py --csvpath annotation.csv --gtpath ./val/R --predpath ./pred/R --output ./results --pncc --psnr --ssim
```

### Outputs (example)
```
running: python psnr_2dirs.py -d0 ./val/T -d1 ./pred -o ./results/psnr.txt
running: python ssim_2dirs.py -d0 ./val/T -d1 ./pred -o ./results/ssim.txt
running: python pncc_2dirs.py -d0 ./val/T -d1 ./pred -o ./results/pncc.txt
combing all metrics and output category-wise results...
180 {'BRBT': 6, 'BRST': 100, 'SRST': 74, 'weak': 67, 'medium': 78, 'strong': 35, 'ghost_yes': 59, 'ghost_no': 121}
      BRBT 006 0.857 0.857 0.857
      BRST 100 0.990 0.990 0.990
      SRST 074 0.987 0.987 0.987
      weak 067 0.985 0.985 0.985
    medium 078 0.987 0.987 0.987
    strong 035 0.972 0.972 0.972
 ghost_yes 059 0.983 0.983 0.983
  ghost_no 121 0.992 0.992 0.992
Done! Benchmarking results are ready!
```

