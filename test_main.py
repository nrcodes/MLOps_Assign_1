import unittest
import numpy as np
import pickle
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample Data
        cls.sample_input = np.array([
            12.47, 18.6, 81.09, 481.9, 0.09965, 0.1058, 0.08005, 0.03821, 0.1925, 0.06373,
            0.3961, 1.044, 2.497, 30.29, 0.006953, 0.01911, 0.02701, 0.01037, 0.01782, 0.003586,
            14.97, 24.64, 96.05, 677.9, 0.1426, 0.2378, 0.2671, 0.1015, 0.3014, 0.0875
        ]).reshape(1, -1)
        
        with open("models/LR_model.pkl", "rb") as lr:
            cls.logistic_regression = pickle.load(lr)
        with open("models/RF_model.pkl", "rb") as rf:
            cls.random_forest = pickle.load(rf)
        with open("models/SVM_model.pkl", "rb") as s_v_m:
            cls.svm = pickle.load(s_v_m)

    def get_prediction_message(self, prediction):
        return "The patient has benign tumor" if prediction == 0 else "The patient has malignant tumor"

    def test_logistic_regression(self):
        data = self.sample_input
        prediction = self.logistic_regression.predict(data)[0]
        prediction_message = self.get_prediction_message(prediction)
        self.assertIn(prediction_message, ["The patient has benign tumor", "The patient has malignant tumor"],
                      "Logistic Regression prediction message is incorrect")
        logging.info("Logistic Regression test case passed")

    def test_random_forest(self):
        data = self.sample_input
        prediction = self.random_forest.predict(data)[0]
        prediction_message = self.get_prediction_message(prediction)
        self.assertIn(prediction_message, ["The patient has benign tumor", "The patient has malignant tumor"],
                      "Random Forest prediction message is incorrect")
        logging.info("Random Forest test case passed")

    def test_svm(self):
        data = self.sample_input
        prediction = self.svm.predict(data)[0]
        prediction_message = self.get_prediction_message(prediction)
        self.assertIn(prediction_message, ["The patient has benign tumor", "The patient has malignant tumor"],
                      "SVM prediction message is incorrect")
        logging.info("SVM test case passed")

if __name__ == '__main__':
    unittest.main()
