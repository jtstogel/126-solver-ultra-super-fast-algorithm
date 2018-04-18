rm fast_transition_matrix.*
cd hittingtime_numpyextension
python setup.py build_ext --inplace
mv fast_transition_matrix.* ../
rm -r build
cd ../
