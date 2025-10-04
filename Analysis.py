import numpy as np
from termcolor import cprint, colored
from Sub_Functions.Load_data import models_return_metrics, Load_data2, Load_data, train_test_splitter, \
    train_test_splitter1
# from Proposed_model.Proposed_model import proposed_model
from Proposed_model.Proposed_model import proposed_model
class Analysis:
    def __init__(self, Data):
        self.lab = None
        self.feat = None
        self.DB = Data
        self.E = [20, 40, 60, 80, 100]
        self.save = False

    def Data_loading(self):
        self.feat = np.load(f"data_loader/{self.DB}/features.npy")
        # loading the labels
        self.lab = np.load(f"data_loader/{self.DB}/labels.npy")

    def COMP_Analysis(self):
        self.Data_loading()
        tr = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        models_return_metrics(self.DB, ok=True, epochs=100)

        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR", "PC1", "RE1"]
        files_name = [f"Analysis/Comparative_Analysis/{self.DB}/{name}_1.npy" for name in perf_names]

        A = np.load(f"Temp/{self.DB}/Comp/ATT_CNN/all_metrics.npy", allow_pickle=True).tolist()
        B = np.load(f"Temp/{self.DB}/Comp/CAE/all_metrics.npy", allow_pickle=True).tolist()
        C = np.load(f"Temp/{self.DB}/Comp/dCNN/all_metrics.npy", allow_pickle=True).tolist()
        D = np.load(f"Temp/{self.DB}/Comp/InceptionV3/all_metrics.npy", allow_pickle=True).tolist()
        E = np.load(f"Temp/{self.DB}/Comp/MPFCNN/all_metrics.npy", allow_pickle=True).tolist()
        F = np.load(f"Temp/{self.DB}/Comp/VGGG16_KNN/all_metrics.npy", allow_pickle=True).tolist()
        G = np.load(f"Temp/{self.DB}/Comp/ViT_CA/all_metrics.npy", allow_pickle=True).tolist()

        all_models = [A, B, C, D, E, F, G]

        if self.save:
            for j in range(len(perf_names)):
                new = []
                for model_metrics in all_models:
                    x = [row[j] for row in model_metrics]
                    new.append(x)
                if self.save:
                    np.save(files_name[j], np.array(new))

    def PERF_Analysis(self):
        epoch = [0]
        Performance_Results = []
        Training_Percentage = 0.4
        print("The performance Analysis starts....")

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count  Is {i + 1} Out Of 6", 'cyan', on_color='on_grey')
            output = []
            for ep in epoch:
                x_train, x_test, y_train, y_test = train_test_splitter1(self.DB, percent=Training_Percentage)
                result = proposed_model(x_train, x_test, y_train, y_test, Training_Percentage, self.DB)
                output.append(result)
            Performance_Results.append(output)

            Training_Percentage += 0.1

        cprint("The results are saved successfully")
        cprint("[✅] Execution of Performance Analysis Completed", 'green', on_color='on_grey')
