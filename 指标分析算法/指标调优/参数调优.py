from 指标对比 import zhibiao
diqv = "D1"
# "五一假期平均小时地铁出站量","五月五调休平均小时出站量","5.5-5.10调休平均小时出站量","周末平均小时出站量"
time ="五一假期平均小时地铁出站量"
Alllist = []
print("开始")
for i in [0.2,0.3,0.4,0.5]:
    for j in range(1,100):
        test_size,random_state,R2_improvement= zhibiao(i,j,diqv,time)
        if R2_improvement == 0:
            continue
        element = {
            "test_size":test_size,
            "random_state":random_state,
            "R2_improvement":R2_improvement,
        }
        Alllist.append(element)
if len(Alllist) == 0:
    print("没有满足条件的字典")
    print("结束")
else:
    print("Alllist长度:", len(Alllist))
    max_item = max(Alllist, key=lambda d: d["R2_improvement"])
    print("最大 R2_improvement的字典:", max_item)
    max_R2_improvement = max_item["R2_improvement"]
    print("最大 R2_improvement:", max_R2_improvement)
    zhibiao(max_item["test_size"],max_item["random_state"],diqv,time)
    print("结束")