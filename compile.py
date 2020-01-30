from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("bank_form_process",  ["bank_form_process.py"]),
    Extension("remove_lines", ["remove_lines.py"]),
    Extension("create_area", ["create_area.py"]),
    Extension("deskew", ["deskew.py"]),
#   ... all your modules that need be compiled ...
]
setup(
    name = 'Bank Form Process',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
