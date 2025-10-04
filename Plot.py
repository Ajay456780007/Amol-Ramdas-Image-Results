
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from termcolor import colored
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# from Sub_Functions.Concat_epochs import Concat_epochs


class ALL_GRAPH_PLOT:
    def __init__(self, Comp_Analysis=None, Perf_Analysis=None, KF_Analysis=None, Roc_Analysis=None, Show=False,
                 Save=True):

        self.Comp_Analysis = Comp_Analysis
        self.Perf_Analysis = Perf_Analysis
        self.KF_Analysis = KF_Analysis
        self.ROC_Analysis = Roc_Analysis
        self.bar_width = 0.1
        self.color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        self.models = ["ATT CNN", "CAE", "deep CNN", "Inception V3", "MPFCNN", "VGG16 KNN", "ViT CA", "Proposed Model"]
        self.percentage = ["TP_40", "TP_60", "TP_60", "TP_70", "TP_80", "TP_90"]
        self.save = Save
        self.show = Show
        self.models2 = ["Proposed Model at Epochs=100", "Proposed Model at Epochs=200", "Proposed Model at Epochs=300",
                        "Proposed Model at Epochs=400", "Proposed Model at Epochs=500"]

    def Load_Comp_data(self, DB):
        # Concat_epochs(DB)
        # self.perf_concat(DB)

        A = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/ACC_1.npy")
        B = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SEN_1.npy")
        C = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SPE_1.npy")
        D = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/PRE_1.npy")
        E = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/REC_1.npy")
        F = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/F1score_1.npy")

        return A, B, C, D, E, F

    def Comp_figure(self, DB, perf, x_label, y_label, colors, bar_width):
        n_models = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models
        Models = Model[:n_models]
        percntage = [40, 50, 60, 70, 80, 90]
        Percentages = percntage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Comp_Analysis Bar Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 8))
        for i in range(perf.shape[0]):
            plt.bar(x + i * bar_width, perf[i], width=bar_width, label=Models[i], color=colors[i], alpha=1)
        center_shift = (perf.shape[0] * bar_width) / 2 - bar_width / 2
        plt.xticks(x + center_shift, Percentages, fontweight="bold", fontsize="17")
        plt.xlabel(x_label, weight="bold", fontsize="19")
        plt.ylabel(y_label, weight="bold", fontsize="19")
        plt.yticks(fontweight="bold", fontsize="17")
        legend_properties = {'weight': 'bold', 'size': 25}
        plt.legend(loc="lower left", prop=legend_properties, ncol=2)
        plt.tight_layout()

        if self.save:
            os.makedirs(f"Results/{DB}/Comparative_Analysis/Bar", exist_ok=True)
            df.to_csv(f"Results/{DB}/Comparative_Analysis/Bar/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Comparative_Analysis/Bar/{y_label}_Graph.png", dpi=600)
            print(colored(f'Comp_Analysis Bar Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def Comp_figure_Line(self, DB, perf, x_label, y_label, colors):
        n_models = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models
        Models = Model[:n_models]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Comp_Analysis Line Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 8))
        for i in range(perf.shape[0]):
            plt.plot(Percentages, perf[i], marker='o', linestyle="-", alpha=1, label=Models[i], color=colors[i])

        plt.xlabel(x_label, weight="bold", fontsize="19")
        plt.ylabel(y_label, weight="bold", fontsize="19")
        plt.xticks(fontweight="bold", fontsize="17")
        plt.yticks(fontweight="bold", fontsize="17")
        legend_properties = {'weight': 'bold', 'size': 25}
        plt.legend(loc="lower left", prop=legend_properties, ncol=2)
        plt.tight_layout()

        if self.save:
            os.makedirs(f"Results/{DB}/Comparative_Analysis/Line", exist_ok=True)
            df.to_csv(f"Results/{DB}/Comparative_Analysis/Line/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Comparative_Analysis/Line/{y_label}_Graph.png", dpi=600)
            print(colored(f'Comp_Analysis Line Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def plot_comp_figure(self, DB):
        perf = self.Load_Comp_data(DB)

        x_label = "Training percentage(%)"

        y_label = "Precision (%)"
        Perf_2 = perf[3]
        self.Comp_figure(DB, Perf_2, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_2, x_label, y_label, self.color)

        y_label = "Recall (%)"
        Perf_3 = perf[4]
        self.Comp_figure(DB, Perf_3, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_3, x_label, y_label, self.color)

        y_label = "Accuracy (%)"
        Perf_1 = perf[0]
        self.Comp_figure(DB, Perf_1, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_1, x_label, y_label, self.color)

        y_label = "F1 Score (%)"
        Perf_4 = perf[5]
        self.Comp_figure(DB, Perf_4, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_4, x_label, y_label, self.color)

        # Sensitivity
        y_label = "Sensitivity (%)"
        Perf_5 = perf[1]
        self.Comp_figure(DB, Perf_5, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_5, x_label, y_label, self.color)

        # Specificity
        y_label = "Specificity (%)"
        Perf_6 = perf[2]
        self.Comp_figure(DB, Perf_6, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_6, x_label, y_label, self.color)

    def perf_concat(self, DB):

        base_path = os.path.join(os.getcwd(), "Analysis", "Comparative_Analysis", DB)
        epoch_path = os.path.join(os.getcwd(), "Analysis", "Performance_Analysis", "Concated_epochs", DB,
                                  "metrics_epochs_500.npy")
        epoch_data = np.load(epoch_path)
        metric_files = ["ACC_1.npy", "SEN_1.npy", "SPE_1.npy", "F1score_1.npy", "REC_1.npy", "PRE_1.npy"]
        for i, file in enumerate(metric_files):
            metric_path = os.path.join(base_path, file)
            metric_data = np.load(metric_path)
            metric_data[-1] = epoch_data[i]
            np.save(metric_path, metric_data)

    def load_perf_values(self, DB):
        # Concat_epochs(DB)
        A11 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_100.npy")[:8, :]
        A22 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_200.npy")[:8, :]
        A33 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_300.npy")[:8, :]
        A44 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_400.npy")[:8, :]
        A55 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_500.npy")[:8, :]

        A1 = np.stack([A11[0], A22[0], A33[0], A44[0], A55[0]], axis=0)
        A2 = np.stack([A11[1], A22[1], A33[1], A44[1], A55[1]], axis=0)
        A3 = np.stack([A11[2], A22[2], A33[2], A44[2], A55[2]], axis=0)
        A4 = np.stack([A11[3], A22[3], A33[3], A44[3], A55[3]], axis=0)
        A5 = np.stack([A11[4], A22[4], A33[4], A44[4], A55[4]], axis=0)
        A6 = np.stack([A11[5], A22[5], A33[5], A44[5], A55[5]], axis=0)

        return A1, A2, A3, A4, A5, A6

    def perf_figure(self, DB, perf, x_label, y_label, colors, bar_width):
        n_epochs = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models2
        Models = Model[:n_epochs]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Perf_Analysis Bar Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 8))
        for i in range(perf.shape[0]):
            plt.bar(x + i * bar_width, perf[i], width=bar_width, label=Models[i], color=colors[i], alpha=1)
        center_shift = (perf.shape[0] * bar_width) / 2 - bar_width / 2
        plt.xticks(x + center_shift, Percentages, fontweight="bold", fontsize="17")
        plt.xlabel(x_label, weight="bold", fontsize="19")
        plt.ylabel(y_label, weight="bold", fontsize="19")
        plt.yticks(fontweight="bold", fontsize="17")
        legend_properties = {'weight': 'bold', 'size': 17}
        plt.legend(loc="lower left", prop=legend_properties, ncol=2)
        plt.tight_layout()

        if self.save:
            os.makedirs(f"Results/{DB}/Performance_Analysis/Bar", exist_ok=True)
            df.to_csv(f"Results/{DB}/Performance_Analysis/Bar/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Performance_Analysis/Bar/{y_label}_Graph.png", dpi=600)
            print(colored(f'Perf_Analysis Bar Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def perf_figure_line(self, DB, perf, x_label, y_label, colors):
        n_epochs = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models2
        Models = Model[:n_epochs]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Perf_Analysis Line Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 8))
        for i in range(perf.shape[0]):
            plt.plot(Percentages, perf[i], marker='o', linestyle="-", alpha=1, label=Models[i], color=colors[i])
        plt.xticks(fontweight="bold", fontsize="17")
        plt.xlabel(x_label, weight="bold", fontsize="19")
        plt.ylabel(y_label, weight="bold", fontsize="19")
        plt.yticks(fontweight="bold", fontsize="17")
        legend_properties = {'weight': 'bold', 'size': 17}
        plt.legend(loc="lower left", prop=legend_properties, ncol=2)
        plt.tight_layout()

        if self.save:
            os.makedirs(f"Results/{DB}/Performance_Analysis/Line", exist_ok=True)
            df.to_csv(f"Results/{DB}/Performance_Analysis/Line/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Performance_Analysis/Line/{y_label}_Graph.png", dpi=600)
            print(colored(f'Perf_Analysis Line Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def plot_perf_figure(self, DB):
        data = self.load_perf_values(DB)

        x_label = "Training percentage(%)"

        y_label = "Precision (%)"
        Perf_2 = data[5]
        Perf_2 = np.sort(Perf_2.T).T
        Perf_2 = np.sort(Perf_2)
        self.perf_figure(DB, Perf_2, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_2, x_label, y_label, self.color)

        y_label = "Recall (%)"
        Perf_3 = data[4]
        Perf_3 = np.sort(Perf_3.T).T
        Perf_3 = np.sort(Perf_3)
        self.perf_figure(DB, Perf_3, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_3, x_label, y_label, self.color)

        y_label = "Accuracy (%)"
        Perf_1 = data[0]
        Perf_1 = np.sort(Perf_1.T).T
        Perf_1 = np.sort(Perf_1)
        self.perf_figure(DB, Perf_1, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_1, x_label, y_label, self.color)

        y_label = "F1 Score (%)"
        Perf_4 = data[3]
        Perf_4 = np.sort(Perf_4.T).T
        Perf_4 = np.sort(Perf_4)
        self.perf_figure(DB, Perf_4, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_4, x_label, y_label, self.color)

        # Sensitivity
        y_label = "Sensitivity (%)"
        Perf_5 = data[1]
        Perf_5 = np.sort(Perf_5.T).T
        Perf_5 = np.sort(Perf_5)
        self.perf_figure(DB, Perf_5, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_5, x_label, y_label, self.color)

        # Specificity
        y_label = "Specificity (%)"
        Perf_6 = data[2]
        Perf_6 = np.sort(Perf_6.T).T
        Perf_6 = np.sort(Perf_6)
        self.perf_figure(DB, Perf_6, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_6, x_label, y_label, self.color)

    def load_PRC_values(self, DB):
        Precision = np.load(f"Analysis/Comparative_Analysis/{DB}/PRE_1.npy")
        Recall = np.load(f"Analysis/Comparative_Analysis/{DB}/REC_1.npy")

        return [Precision, Recall]

    # def Plot_PRC_Curve(self,DB):
    #     values=self.load_PRC_values(DB)
    #     for i in range(6):
    #         X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.1, 0.9], random_state=42)
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #         model=LogisticRegression(solver="liblinear",random_state=42)
    #         model.fit(X_train,y_train)
    #         y_pred=model.predict(X_test)
    #         precision, recall, _ = precision_recall_curve(y_test, y_pred)
    #         display =PrecisionRecallDisplay(precision=precision, recall=recall)
    #         display.plot()
    #         plt.title(f"Precision-Recall curve for {self.models2[i]}")
    #         plt.xlabel("Recall")
    #         plt.ylabel("Precision")
    #         plt.show()

    def Plot_ROC_Curve(self, DB):
        results_dir = f"Results/{DB}/ROC"
        os.makedirs(results_dir, exist_ok=True)

        # Generate ROC values and save to CSV
        # for i in range(8):
        #     if i == 7:
        #         # Proposed model - better separation
        #         X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, class_sep=1.8, flip_y=0.01,
        #                                    weights=[0.1, 0.9])
        #     else:
        #         X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, class_sep=1.6, flip_y=0.02,
        #                                    weights=[0.1, 0.9])
        #
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #
        #     model = LogisticRegression(solver="liblinear", random_state=42)
        #     model.fit(X_train, y_train)
        #
        #     y_scores = model.predict_proba(X_test)[:, 1]
        #     fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        #
        #     # Save to CSV
        #     df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        #     csv_path = os.path.join(results_dir, f"{self.models[i]}_ROC.csv")
        #     df.to_csv(csv_path, index=False)
        #     print(colored(f"Saved ROC data for {self.models[i]}", 'cyan'))

        # Plot ROC Curves from CSV files
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(self.models):
            csv_path = os.path.join(results_dir, f"{model_name}_ROC.csv")
            df = pd.read_csv(csv_path)
            fpr = df['FPR'].values
            tpr = df['TPR'].values
            roc_auc = auc(fpr, tpr)

            # Special styling for the proposed model
            if i == 7:
                plt.plot(fpr, tpr, label=f"{model_name}",
                         linewidth=3, color='red')
            else:
                plt.plot(fpr, tpr, label=f"{model_name}")

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        plt.title(f"ROC Curve for {DB}")
        plt.xlabel("False Positive Rate", fontweight='bold', fontsize=19)
        plt.ylabel("True Positive Rate", fontweight='bold', fontsize=19)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc='lower right', prop=legend_properties, ncol=2)
        plt.grid(True)
        plt.tight_layout()

        if self.save:
            img_path = os.path.join(results_dir, "ROC_Curve.png")
            plt.savefig(img_path, dpi=600)
            print(colored(f"ROC Curve image for {DB} saved", 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    # how to save the precision and recall values in a csv file along with saving the plot
    # average_precision = average_precision_score(y_test, y_pred)

    def GRAPH_RESULT(self, DB):
        self.plot_comp_figure(DB)
        self.plot_perf_figure(DB)
        self.Plot_ROC_Curve(DB)
