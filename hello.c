#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Module method definitions
static PyObject* hello_world_c(PyObject *self, PyObject *args) {
    printf("Hello, world!\n");
    Py_RETURN_NONE;
}

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


static PyObject* hello_numpy_c(PyObject *dummy, PyObject *args)
{
//    printf("in!!!!!!!!!!!!!!!\n");
    PyObject *resourcesArg=NULL;
    PyObject *includeArg;
    PyObject *resultArg;

    PyObject *resourcesArr;
    PyObject *includeArr;
    PyObject *resultArr;
    
    
    double probabilities[11] = {1.0,2.0,3.0,4.0,5.0,6.0,5.0,4.0,3.0,2.0,1.0};
    for (int i = 0; i < 11; i++) {
        probabilities[i] = probabilities[i] / 36.0;
    }

    if (!PyArg_ParseTuple(args, "OOO", &resourcesArg, &includeArg, &resultArg))
        return NULL;

    resourcesArr = PyArray_FROM_OTF(resourcesArg, NPY_DOUBLE, NPY_IN_ARRAY);
    includeArr   = PyArray_FROM_OTF(includeArg, NPY_DOUBLE, NPY_IN_ARRAY);
    resultArr    = PyArray_FROM_OTF(resultArg, NPY_DOUBLE, NPY_INOUT_ARRAY);
    double *resources = PyArray_DATA(resourcesArr);
    double *include   = PyArray_DATA(includeArr);
    double *result    = PyArray_DATA(resultArr);
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
            
            //wn = wn <= 6 ? wn : 6;
            //bn = bn <= 6 ? bn : 6;
            //gn = gn <= 6 ? gn : 6;
            encode(&eTo, wn, bn, gn);
            result[eFrom*343+eTo] += probabilities[k];
        }
    }

    Py_DECREF(resourcesArr);
    Py_DECREF(includeArr);
    Py_DECREF(resultArr);

    return PyInt_FromLong(0);
}


static PyMethodDef hello_methods[] = {
        {
                "hello_python", hello_world_c, METH_VARARGS,
                "Print 'hello xxx'"
        },
        {
                "hello_numpy", hello_numpy_c, METH_VARARGS,
                "numpy function tester",
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef hello_definition = {
        PyModuleDef_HEAD_INIT,
        "hello",
        "A Python module that prints 'hello world' from C code.",
        -1,
        hello_methods
};


PyMODINIT_FUNC PyInit_hello(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&hello_definition);
}
