#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Module method definitions
void decode(int e, int *w, int *b, int *g) {
    *w = e / 49;
    int x = e % 49;
    *b = x / 7;
    *g = x % 7;
//    printf("%d -> %d %d %d\n", e, *w, *b, *g);
}

void encode(int *e, int w, int b, int g) {
    *e = w * 49 + b * 7 + g;
//    printf("%d %d %d -> %d\n", *e);
}

static PyObject *my_callback = NULL;

static PyObject *
set_trade_rule(PyObject *dummy, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(my_callback);  /* Dispose of previous callback */
        my_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}


void call_trade_rule(int w, int b, int g, int *wn, int *bn, int *gn) {
    PyObject *arglist;
    PyObject *result;
    arglist = Py_BuildValue("iii", w, b, g);
    result = PyEval_CallObject(my_callback, arglist);
    if (!PyArg_ParseTuple(result, "iii", wn, bn, gn))
        return;
    Py_DECREF(arglist);
}



static PyObject* populate_transition_matrix(PyObject *dummy, PyObject *args)
{
    PyObject *resourcesArg=NULL;
    PyObject *includeArg;
    PyObject *resultArg;
    PyObject *tradeRuleArg=NULL;

    PyObject *resourcesArr;
    PyObject *includeArr;
    PyObject *resultArr;
    PyObject *tradeRuleArr;
    
    
    double probabilities[11] = {1.0,2.0,3.0,4.0,5.0,6.0,5.0,4.0,3.0,2.0,1.0};
    for (int i = 0; i < 11; i++) {
        probabilities[i] = probabilities[i] / 36.0;
    }

    if (!PyArg_ParseTuple(args, "OOOO", &resourcesArg, &includeArg, &resultArg, &tradeRuleArg))
        return NULL;

    resourcesArr = PyArray_FROM_OTF(resourcesArg, NPY_DOUBLE, NPY_IN_ARRAY);
    includeArr   = PyArray_FROM_OTF(includeArg, NPY_DOUBLE, NPY_IN_ARRAY);
    resultArr    = PyArray_FROM_OTF(resultArg, NPY_DOUBLE, NPY_INOUT_ARRAY);
    tradeRuleArr = PyArray_FROM_OTF(tradeRuleArg, NPY_DOUBLE, NPY_IN_ARRAY);

    double *resources = PyArray_DATA(resourcesArr);
    double *include   = PyArray_DATA(includeArr);
    double *result    = PyArray_DATA(resultArr);
    double *trade_rule= PyArray_DATA(tradeRuleArr);
    /*
     * my code starts here
     */
    npy_intp *shape = PyArray_DIMS(includeArr);
    int numToInclude = *shape; // One dimensional array, so just get first element of shape
    int w,b,g, wn, bn, gn, eFrom, eTo;
    for (int i = 0; i<numToInclude; i++) {
        eFrom = (int) include[i];
        decode(eFrom, &w, &b, &g);
        for (int k = 0; k < 11; k++) {
            wn = w + resources[3*k];
            wn = wn <= 6 ? wn : 6;
            bn = b + resources[3*k+1];
            bn = bn <= 6 ? bn : 6;
            gn = g + resources[3*k+2];
            gn = gn <= 6 ? gn : 6;
            encode(&eTo, wn, bn, gn);
            if (eTo >= 343) {
                PyErr_SetString(PyExc_ValueError, "Ooops.");
                return NULL;
            }
            eTo = (int) trade_rule[eTo];
            result[eFrom*343+eTo] += probabilities[k];
        }
    }

    Py_DECREF(resourcesArr);
    Py_DECREF(includeArr);
    Py_DECREF(resultArr);
    Py_DECREF(tradeRuleArr);

    return PyInt_FromLong(0);
}


static PyMethodDef fast_transition_matrix_methods[] = {
        {
                "populate_transition_matrix", populate_transition_matrix, METH_VARARGS,
                "populates the transition matrix",
        },
	    {
		    "set_trade_rule", set_trade_rule, METH_VARARGS,
		    "set the trade_rule function",
	    },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef fast_transition_matrix_definition = {
        PyModuleDef_HEAD_INIT,
        "fast_transition_matrix",
        "A Python module that populates a transition matrix according to a trading rule.",
        -1,
        fast_transition_matrix_methods
};


PyMODINIT_FUNC PyInit_fast_transition_matrix(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&fast_transition_matrix_definition);
}
