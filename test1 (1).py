import csv
import pprint
import math
import numpy
import numpy as np
from numpy import size
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import matplotlib as mpl

from numpy.random import normal
from scipy.stats import lognorm, poisson
#import tensorflow_probability as tfp

with open('/Users/ashwinxaviourwilliam/Downloads/SCC.444_Programming_assignment_21-22_data_set_clean.csv',newline='') as csvfile:
    df = csv.DictReader(csvfile)
list_loss_sum = []
list_lost_cnt = []
list_loss=[]
list_pert=[]
log_list=[]
error_list=[]
lgmean_list=[]
lgsig=[]
mn=[]
md=[]
mod=[]
vr=[]
sd=[]
pois=[]
pf=[]
impls=[]
list_lognorm=[]
ale=[]
def ercal():
    with open('/Users/ashwinxaviourwilliam/Downloads/SCC.444_Programming_assignment_21-22_data_set_clean.csv',newline='') as csvfile:
        df = csv.DictReader(csvfile)
        for row in df:
            try:
                pr = ((0 <= float(row["prob_freq"])))
                lb = ((0 <= float(row["lb_loss"])))
                lbub = ((float(row["lb_loss"]) > float(row["ub_loss"])))
                if row["prob_freq"] == "":
                    error_list.append(str(row["prob_freq"]) + str(row["risk_name"]))
                if row["lb_loss"] == "":
                    error_list.append(str(row["lb_loss"]) + str(row["risk_name"]))
                if row["ub_loss"] == "":
                    error_list.append(str(row["ub_loss"]) + str(row["risk_name"]))
                if float(lb & lbub):
                    error_list.append("Error value _" + str(row["lb_loss"]) + " ___" + str(row["ub_loss"]))
                if pr:
                    error_list.append(pr)
            except ValueError:
                print(
                    str(row) + " " + "contains string value, which is not acceptable. It must be a valid number of integer type")
def lono():
    with open('/Users/ashwinxaviourwilliam/Downloads/SCC.444_Programming_assignment_21-22_data_set_clean.csv',newline='') as csvfile:
        df = csv.DictReader(csvfile)
        for row in df:
            mypoission=poisson(int(row['prob_freq']))
            row["men"]= mypoission.mean()
            mn.append(row["men"])
            row["med"]=mypoission.median()
            md.append(row["med"])
            #row["nmed"]=mypoission.mean() + (1/3) -(0.02/ mypoission.mean())
            #print("new median",row["nmed"])
            row["var"]= mypoission.var()
            vr.append(row["var"])
            row["sd"]=mypoission.std()
            sd.append(row["sd"])
            #row["mod"]= mypoission.mode()
            #mod.append(row["mod"])
            pois.append(row)
def nom():
    with open('/Users/ashwinxaviourwilliam/Downloads/SCC.444_Programming_assignment_21-22_data_set_clean.csv',
              newline='') as csvfile:
        df = csv.DictReader(csvfile)
        for row in df:
            row["mu"]=((math.log(float(row["ub_loss"]))) + (math.log(float(row["lb_loss"]))))/2
            lgmean_list.append(row["mu"])
            row["si"]=((math.log(float(row["ub_loss"]))) - (math.log(float(row["lb_loss"]))) )/3.29
            lgsig.append(row["si"])
            row["imptls"]= float(row["prob_freq"]) * ( math.e ** (float(row["mu"] )+ (float((row["si"]) ** 2)/2)))
            impls.append(row["imptls"])
            row["ale"]= (float(row["prob_freq"])) * (float(row["mu"]))
            ale.append(row["ale"])
            list_lognorm.append(row)
def mont():
            rng = numpy.random.default_rng()
  
            for dist in list_lognorm:
                dist['dist'] = lognorm(dist['si'], scale=math.exp(dist['mu']))
            mc_run = 10
            for yr in range(mc_run):
                loss_sum = 0
                loss_event_cnt = 0
                for ln in list_lognorm:
                    r = rng.random()
                    if r <= float(ln['prob_freq']):
                        loss = ln['dist'].rvs(1)[0]
                        loss_event_cnt += 1
                        loss_sum += loss
                        print("event",loss_event_cnt)
                        print("loss", loss)
                        print("sum loss",loss_sum)
                        list_loss.append(loss)
                        list_loss_sum.append(loss_sum)
                        list_lost_cnt.append(loss_event_cnt)

            print("loss sum", list_loss_sum)
            print("list loss count", list_lost_cnt)
            print("the loss", list_loss)
            losses = numpy.array([numpy.percentile(list_loss_sum, x) for x in range(1, 100, 1)])
            percentiles = numpy.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])
            print("------------------------------Task 6------------------------------")
            print("\nSummary")
            for i in range(0, len(list_loss_sum)):
                print(i, list_loss_sum[i], list_lost_cnt[i])
            print("\nLoss Exceedence Statistics")
            print(f"Average Loss in a year is {numpy.mean(list_loss)}")
            print(f"Min loss in a year {numpy.min(list_loss)}")
            print(f"Max loss in a year {numpy.max(list_loss)}")
            print(f"Average number of events in a year {numpy.mean(list_lost_cnt)}")
            print(f"Min number of events per year {numpy.min(list_lost_cnt)}")
            print(f"Max number of events per year {numpy.max(list_lost_cnt)}")
            print(f"There is a {percentiles[25] * 100}% of losses exceeding {losses[25]:.2f} or more")
            print(f"There is a {percentiles[50] * 100}% of losses exceeding {losses[50]:.2f} or more")
            print(f"There is a {percentiles[75] * 100}% of losses exceeding {losses[75]:.2f} or more")

            inv_loss = 5000
            idx = numpy.where(losses >= inv_loss)
            if len(idx[0]) > 0:
                print(f"There is a {percentiles[idx[0][0]] * 100}% of losses exceeding {inv_loss:.2f} or more")
                fig = plt.figure()
                ax = plt.gca()
                ax.plot(losses, percentiles)
                title = "Loss Exceedance  curve"
                xlim = [1, 10000000]
                plt.title(title)
                ax.set_xscale("log")
                ax.set_ylim(0.0, percentiles[numpy.argmax(losses > 0.0)] + 0.05)
                ax.set_xlim(xlim[0], xlim[1])
                xtick = mtick.StrMethodFormatter('£{x:,.0f}')
                ax.xaxis.set_major_formatter(xtick)
                ytick = mtick.StrMethodFormatter('{x:.000%}')
                ax.yaxis.set_major_formatter(ytick)
                plt.grid(which='both')
                plt.show()
if __name__ == "__main__":
    print("------------------------------Task 3------------------------------")
    ercal()
    for items in error_list:
        print(items)
    print("---------------------------Task4---------------------------")
    lono()
    nom()
    print("----------------lognorm mu----------------")
    for it in lgmean_list:
        print(it)
    print("----------------lognorm sig----------------")
    for i in lgsig:
        print(i)
    print("---------------actmean---------------")
    for a in mn:
        print(a)
    print("---------------actmed---------------")
    for a in md:
        print(a)
    print("---------------var---------------")
    for a in vr:
        print(a)
    print("---------------sd---------------")
    for a in sd:
        print(a)
    print("---------------poisson distrubation---------------")
    print(pois)
    print("---------------impact loss---------------")
    print(impls)
    print("---------------log norm list---------------")
    print(list_lognorm)
    print("---------------annual loss exp---------------")
    print(ale)
    print("------------------------------Task 5------------------------------")
    mont()
