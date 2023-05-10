import unittest
import os
import logging
import shutil

import filecmp

import pandas as pd
from pandas.testing import assert_frame_equal

import amcp.modes as modes
from amcp.amcp import argument_parser

class test_amcp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.disable(logging.NOTSET)

    @classmethod
    def tearDownClass(cls):

        pass

    def test_validation_and_train(self):
        """Test to check validation and training method
        """

        args = argument_parser(['-i', 'amcp_test/goldenData/proteinTest_50k_train_amcp.txt.bz2', '-m', 'validation'])

        # VALIDATION TEST

        args.in_file = 'amcp_test/goldenData/proteinTest_50k_train_amcp.txt.bz2'
        args.out_file = 'amcp_test/test_validation_out.txt'
        args.out_file_train = 'amcp_test/test_validation_train_out.txt'
        args.nosmooth = True

        args.numberModels = 2
        args.validationFolds = 2

        modes.validation(args)

        validation_out_golden = 'amcp_test/goldenData/test_validation_out_golden.txt'
        validation_train_out_golden = 'amcp_test/goldenData/test_validation_train_out_golden.txt'

        dataOutFile = pd.read_csv(args.out_file, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True)
        dataOutFileGolden = pd.read_csv(validation_out_golden, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True)

        dataOutTrainFile = pd.read_csv(args.out_file_train, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True, skiprows=[0,1])
        dataOutTrainFileGolden = pd.read_csv(validation_train_out_golden, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True, skiprows=[0,1])

        assert_frame_equal(dataOutFile, dataOutFileGolden, check_exact = False)
        assert_frame_equal(dataOutTrainFile, dataOutTrainFileGolden, check_exact = False)

        shutil.rmtree('catboost_info')
        os.remove(args.out_file)
        os.remove(args.out_file_train)

        # TRAIN TEST

        args.models_dir = 'amcp_test/amcp_models'
        args.classifier = 'catboost'
        args.ratioTestSet = 0.2
        args.justmodel = None

        modes.train(args)

        m0_alpha0_File = 'amcp_test/amcp_models/amcp_cal_alphas0_m0.z'
        m0_alpha0_File_golden = 'amcp_test/goldenData/amcp_cal_alphas0_m0.z'

        m1_alpha0_File = 'amcp_test/amcp_models/amcp_cal_alphas0_m1.z'
        m1_alpha0_File_golden = 'amcp_test/goldenData/amcp_cal_alphas0_m1.z'

        m0_alpha1_File = 'amcp_test/amcp_models/amcp_cal_alphas1_m0.z'
        m0_alpha1_File_golden = 'amcp_test/goldenData/amcp_cal_alphas1_m0.z'

        m1_alpha1_File = 'amcp_test/amcp_models/amcp_cal_alphas1_m1.z'
        m1_alpha1_File_golden = 'amcp_test/goldenData/amcp_cal_alphas1_m1.z'

        self.assertTrue(filecmp.cmp(m0_alpha0_File, m0_alpha0_File_golden, shallow=False), 'Fail in m0_alpha0 file')
        self.assertTrue(filecmp.cmp(m1_alpha0_File, m1_alpha0_File_golden, shallow=False), 'Fail in m1_alpha0 file')
        self.assertTrue(filecmp.cmp(m0_alpha1_File, m0_alpha1_File_golden, shallow=False), 'Fail in m0_alpha1 file')
        self.assertTrue(filecmp.cmp(m1_alpha1_File, m1_alpha1_File_golden, shallow=False), 'Fail in m1_alpha1 file')

        shutil.rmtree('amcp_test/amcp_models')
        shutil.rmtree('catboost_info')


    def test_predict(self):
        """Test to check predict method
        """

        args = argument_parser(['-i', 'amcp_test/goldenData/proteinTest_100k_predict_amcp.txt.bz2', '-m', 'validation'])

        args.models_dir = 'amcp_test/goldenData'
        args.out_file = 'amcp_test/test_predict_100k_out.txt'

        modes.predict(args)

        out_file_bz2 = args.out_file + '.bz2'
        predict_out_golden = 'amcp_test/goldenData/test_predict_100k_out.txt.bz2'
    
        dataOutFile = pd.read_csv(out_file_bz2, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True, skiprows=[0,1])
        dataOutFileGolden = pd.read_csv(predict_out_golden, header=None, names = ['molID', 'real', 'p0', 'p1'], delim_whitespace=True, skiprows=[0,1])

        assert_frame_equal(dataOutFile, dataOutFileGolden, check_exact = False)

        os.remove(out_file_bz2)


if __name__ == '__main__'  and __package__ is None:

    unittest.main()