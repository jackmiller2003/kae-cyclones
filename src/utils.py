from experiment import *

def weight_decay():
    # wd=0.01
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_2, _ = ge_exp.run(epochs=20, wd=0.01)
    # wd=0.1
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_3, _ = ge_exp.run(epochs=20, wd=0.1)
    # wd=1
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss_4, _ = ge_exp.run(epochs=20, wd=1.0)
    # our method
    eigen_exp = Experiment("origin_mse", "gaussianEigen", 1.0, "ocean")
    eigen_loss, _ = eigen_exp.run(epochs=20)
    
    
    colors = ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07']
    epochs = [x for x in range(len(ge_loss_2["fwd_val"]))]
    #plt.plot(epochs, ge_loss_1["fwd_val"], label=r"$\alpha=0.0$", color=colors[0])
    plt.plot(epochs, ge_loss_2["fwd_val"], label=r"$\alpha=0.01$", color=colors[1])
    plt.plot(epochs, ge_loss_3["fwd_val"], label=r"$\alpha=0.1$", color=colors[2])
    plt.plot(epochs, ge_loss_4["fwd_val"], label=r"$\alpha=1.0$", color=colors[3])
    plt.plot(epochs, eigen_loss["fwd_val"], label="Eigenloss (Origin MSE)", color=colors[4])
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.legend(loc="best")
    save_path = "/home/156/cn1951/kae-cyclones/results/images/"
    plt.savefig(save_path+"weight_decay.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)


def wall_time():
    # GAUSSIAN ELEMENT
    ge_exp = Experiment("none", "gaussianElement", 2.5, "ocean")
    ge_loss, ge_time = ge_exp.run(epochs=20, batchSize=124, return_time=True)
    # GAUSSIAN EIGEN
    eigen_exp = Experiment("origin_mse", "gaussianEigen", 1.0, "ocean")
    eigen_loss, eigen_time = eigen_exp.run(epochs=20, return_time=True)
    # UNIFORM EIGEN
    unif_exp = Experiment("origin_mse", "gaussianElement", 1.0, "ocean")
    unif_loss, unif_time = unif_exp.run(epochs=20, return_time=True)
    # UNIT PERTURB
    unit_exp = Experiment("none", "gaussianElement", 1.0, "ocean")
    unit_loss, unit_time = unit_exp.run(epochs=20, return_time=True)
    
    colors = ['black', '#003DFD', '#b512b8', '#11a9ba', '#0d780f', '#f77f07']
    plt.plot(ge_time, ge_loss["fwd_val"], label="No penalty", color=colors[0])
    plt.plot(unif_time, unif_loss["fwd_val"], label="Eigenloss only", color=colors[1])
    plt.plot(unit_time, unit_loss["fwd_val"], label="Eigeninit only", color=colors[2])
    plt.plot(eigen_time, eigen_loss["fwd_val"], label="Eigenloss and eigeninit", color=colors[3])
    plt.xlim(0, 30)
    plt.xlabel("Wall time (seconds)")
    plt.ylabel("Validation loss")
    plt.legend(loc="best")
    save_path = "/home/156/cn1951/kae-cyclones/results/images/"
    plt.savefig(save_path+"test_time.pdf", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)  