import argparse, math, random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

#Reading the .txt data file
def read_data(path):
    cities=[]
    for line in open(path):
        parts=line.strip().split()
        if len(parts)>=3 and parts[0].isdigit():
            cities.append((float(parts[1]),float(parts[2])))
    return cities

#Euclidean distance calculation between two points
def euclidean(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dist_matrix(cities):
    n=len(cities)
    d=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            d[i][j]=d[j][i]=euclidean(cities[i],cities[j])
    return d

#Computing total distance
def path_len(path,dmat):
    return sum(dmat[path[i]][path[(i+1)%len(path)]] for i in range(len(path)))

def two_opt(path,dmat):
    best=path[:]; bestlen=path_len(best,dmat)
    n=len(best); improved=True
    while improved:
        improved=False
        for i in range(1,n-2):
            for j in range(i+1,n):
                if j-i==1: continue
                new=best[:]
                new[i:j]=best[j-1:i-1:-1]
                newlen=path_len(new,dmat)
                if newlen<bestlen:
                    best,bestlen=new,newlen; improved=True
    return best,bestlen

def draw(cities,path,outfile):
    xs=[cities[i][0] for i in path]+[cities[path[0]][0]]
    ys=[cities[i][1] for i in path]+[cities[path[0]][1]]
    plt.plot(xs,ys,'b-',lw=1)
    plt.scatter([c[0] for c in cities],[c[1] for c in cities],c='red',s=20)
    plt.scatter(cities[path[0]][0],cities[path[0]][1],c='green',s=60,label="Start")
    plt.title("TSP Output")
    plt.savefig(outfile,dpi=200)
    plt.close()

def setup(n):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    tb=base.Toolbox()
    tb.register("indices", random.sample, range(n), n)
    tb.register("individual", tools.initIterate, creator.Individual, tb.indices)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament, tournsize=2)
    tb.register("mate", tools.cxOrdered)
    tb.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    return tb

def main(args):
    cities=read_data(args.data)
    dmat=dist_matrix(cities)
    tb=setup(len(cities))
    tb.register("evaluate", lambda ind:(path_len(ind,dmat),))
    random.seed(args.seed)
    pop=tb.population(n=args.popsize); hof=tools.HallOfFame(1)
    algorithms.eaSimple(pop,tb,args.cxpb,args.mutpb,args.generations,None,hof,False)
    best=hof[0]; length=path_len(best,dmat)
    best,length=two_opt(best,dmat)
    print("Shortest tour length =", round(length,3))
    print("Tour order:", [i+1 for i in best])
    if args.img: draw(cities,best,args.img)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="TSPDATA.txt")
    p.add_argument("--generations",type=int,default=3500)
    p.add_argument("--popsize",type=int,default=800)
    p.add_argument("--cxpb",type=float,default=0.7)
    p.add_argument("--mutpb",type=float,default=0.3)
    p.add_argument("--seed",type=int,default=40)
    p.add_argument("--img",default="tsp.png")
    args=p.parse_args()
    main(args)
