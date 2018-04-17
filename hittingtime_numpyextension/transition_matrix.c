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

// All we take in is a flattened resources array, an array [w, b, g] and an array for the trade rule
// This will return a 2d matrix that needs to be solved, and a list of indexes D st Beta[D[encode(w,b,g)]] is the expected value for starting at (w,b,g)
static PyObject* populate_transition_matrix(PyObject *dummy, PyObject *args)
{
    PyObject *resourcesArg=NULL;
    PyObject *goalArg=NULL;
    PyObject *tradeRuleArg=NULL;

    PyObject *resourcesArr;
    PyObject *goalArr;
    PyObject *tradeRuleArr;
    
    double probabilities[11] = {1.0,2.0,3.0,4.0,5.0,6.0,5.0,4.0,3.0,2.0,1.0};
    for (int i = 0; i < 11; i++) {
        probabilities[i] = probabilities[i] / 36.0;
    }

    if (!PyArg_ParseTuple(args, "OOO", &resourcesArg, &goalArg, &tradeRuleArg))
        return NULL;

    resourcesArr = PyArray_FROM_OTF(resourcesArg, NPY_DOUBLE, NPY_IN_ARRAY);
    goalArr   = PyArray_FROM_OTF(goalArg, NPY_DOUBLE, NPY_IN_ARRAY);
    tradeRuleArr = PyArray_FROM_OTF(tradeRuleArg, NPY_DOUBLE, NPY_IN_ARRAY);
    
    double *resources  = PyArray_DATA(resourcesArr);
    double *goal       = PyArray_DATA(goalArr);
    double *trade_rule = PyArray_DATA(tradeRuleArr);
    
    int w_goal = (int) goal[0];
    int b_goal = (int) goal[1];
    int g_goal = (int) goal[2];
    int *D = calloc(343, sizeof(int));
    int i = 0;
    int w, b, g;
    int e;
    for (w = 0; w < 7; w++) {
        for(b = 0; b < 7; b++) {
            for (g = 0; g < 7; g++) {
                encode(&e, w, b, g);
                if ((w >= w_goal) && (b >= b_goal) && (g >= g_goal)) {
                    D[e] = -1; // We don't get a place in the final array
                } else {
                    D[e] = i; // We get a place in the final array
                    i++;
                }
            }
        }
    }
    int N = i;
    double *P = calloc(N*N, sizeof(double)); // Our transition matrix
    /*
     * my code starts here
     */
    int wn, bn, gn, eFrom, eTo;
    for (eFrom = 0; eFrom < 343; eFrom++) {
        if (D[eFrom] == -1) continue; // Don't do anything with this if it's not in our array
        decode(eFrom, &w, &b, &g);    // Get the corresponding w,b,g

        for (int k = 0; k < 11; k++) {
            wn = w + resources[3*k];
            wn = wn <= 6 ? wn : 6;
            bn = b + resources[3*k+1];
            bn = bn <= 6 ? bn : 6;
            gn = g + resources[3*k+2];
            gn = gn <= 6 ? gn : 6;
            encode(&eTo, wn, bn, gn);
            if (eTo >= 343 || eTo < 0) {
                PyErr_SetString(PyExc_ValueError, "Ooops.");
                return NULL;
            }
            eTo = (int) trade_rule[eTo];
            if (D[eTo] == -1) continue; // It's not in our array, so why bother
            P[D[eFrom]*N+D[eTo]] += probabilities[k];
        }
    }
    // Pack up our P as numpy arraay
    npy_intp Pdims[1] = {N*N};
    PyObject *Pnarray = PyArray_SimpleNewFromData(1, Pdims, NPY_DOUBLE, P);
    PyArray_ENABLEFLAGS((PyArrayObject*)Pnarray, NPY_ARRAY_OWNDATA);


    npy_intp Ddims[1] = {343};
    PyObject *Dnarray = PyArray_SimpleNewFromData(1, Ddims, NPY_INT, D);
    PyArray_ENABLEFLAGS((PyArrayObject*)Dnarray, NPY_ARRAY_OWNDATA);
    
    PyObject *tuple = Py_BuildValue("(OO)", Pnarray, Dnarray);
    
    Py_DECREF(Pnarray); // Py_BuildValue increments reference counter
    Py_DECREF(Dnarray); // ^same reason
    Py_DECREF(resourcesArr);
    Py_DECREF(goalArr);
    Py_DECREF(tradeRuleArr);

    return tuple;
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
