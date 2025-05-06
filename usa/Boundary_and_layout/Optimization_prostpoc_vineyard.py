from topfarm.recorders import TopFarmListRecorder
import matplotlib.pyplot as plt 

# Open the recorder that we simulate in deliveriable 2 
recorder = TopFarmListRecorder().load("/Users/yosephhassan/workspace 2/recordings/optimization_vineyardwind.pkl")
plt.figure()
plt.plot(recorder['counter'], recorder['AEP']/recorder['AEP'][-1])
plt.xlabel('Iterations')
plt.ylabel('AEP/AEP_opt')
plt.title("Vineyardwind1 AEP optimization")
plt.show()