from glob import glob
import csv
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--txt", default="txt/T_PNCC_IBCLN_crop.txt", help="path_to_txt")

ARGS = parser.parse_args()


txt_name = ARGS.txt
csv_name = "../RRRR_csv.csv"

txt_file= open(txt_name, 'r')
csv_file= io.open(csv_name, encoding='utf-8-sig')
rows = csv.DictReader(csv_file)
txt_rows= txt_file.readlines()
metrics = {}
for row in txt_rows:
    print(row)
    contents = row[:-1].split(' ')
    metrics[contents[0]] = contents[1:]

# print(metrics)
# psnrs = {}
# ssims = {}
# nccs ={}
# sis = {}
# exit()
cnts = {'BRBT':0, 'BRST':0, 'SRST':0, 'weak':0, 'medium':0, 'strong':0, 'ghost_yes':0, 'ghost_no':0}
psnrs = {'BRBT':0, 'BRST':0, 'SRST':0, 'weak':0, 'medium':0, 'strong':0, 'ghost_yes':0, 'ghost_no':0}
ssims = {'BRBT':0, 'BRST':0, 'SRST':0, 'weak':0, 'medium':0, 'strong':0, 'ghost_yes':0, 'ghost_no':0}
sis = {'BRBT':0, 'BRST':0, 'SRST':0, 'weak':0, 'medium':0, 'strong':0, 'ghost_yes':0, 'ghost_no':0}
nccs = {'BRBT':0, 'BRST':0, 'SRST':0, 'weak':0, 'medium':0, 'strong':0, 'ghost_yes':0, 'ghost_no':0}

cnt = 0
for row in rows:
    # print(row['test'], row['name'], str(row['ghost']), row['reflection'], row['type'], len(row))
    print("ghost_" + str(row['ghost']))
    ghost_name = "ghost_" + str(row['ghost'])
    new_name = row['name'][:1] + '_' +  row['name'].split('/')[-1].replace("vis_","").replace(".json","_T.png")
    if new_name in metrics:
        print(new_name, metrics[new_name])
        cnts[row['type']] += 1
        cnts[row['reflection']] += 1
        if ghost_name == "ghost_yes" or  ghost_name == "ghost_no":
            cnts[ghost_name] += 1
            psnrs[ghost_name] += float(metrics[new_name][0])
            ssims[ghost_name] += float(metrics[new_name][1])
            nccs[ghost_name] += float(metrics[new_name][2])
            sis[ghost_name] += float(metrics[new_name][3])


        psnrs[row['type']] += float(metrics[new_name][0])
        psnrs[row['reflection']] += float(metrics[new_name][0])
        ssims[row['type']] += float(metrics[new_name][1])
        ssims[row['reflection']] += float(metrics[new_name][1])
        nccs[row['type']] += float(metrics[new_name][2])
        nccs[row['reflection']] += float(metrics[new_name][2])
        sis[row['type']] += float(metrics[new_name][3])
        sis[row['reflection']] += float(metrics[new_name][3])

        cnt += 1
    # else:
    #     continue

print(cnt,cnts)
print(ARGS.txt)
for key in cnts:
    print("%10s %03d %.2f %.3f %.3f %.3f" % (key, cnts[key], psnrs[key]/float(cnts[key]+1),  ssims[key]/float(cnts[key]+1), nccs[key]/float(cnts[key]+1), sis[key]/float(cnts[key]+1)))
# print(rows[0])
