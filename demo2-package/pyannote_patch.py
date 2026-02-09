"""
Patch torchaudio before importing pyannote.audio
This fixes compatibility with torchaudio 2.9+ which removed list_audio_backends and AudioMetaData
Also patches hyperpyyaml for Python 3.14 compatibility (ast.Num -> ast.Constant)
"""
import torchaudio
import ast
import sys

# Patch list_audio_backends if it doesn't exist
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        """Return available audio backends"""
        # Try to detect available backends
        backends = []
        try:
            import soundfile
            backends.append("soundfile")
        except ImportError:
            pass
        # Default fallback
        if not backends:
            backends = ["soundfile"]  # Assume soundfile is available
        return backends
    
    torchaudio.list_audio_backends = _list_audio_backends

# Patch AudioMetaData if it doesn't exist (removed in torchaudio 2.9+)
if not hasattr(torchaudio, "AudioMetaData"):
    from collections import namedtuple
    # Create a simple AudioMetaData class
    AudioMetaData = namedtuple('AudioMetaData', ['sample_rate', 'num_frames', 'num_channels'])
    torchaudio.AudioMetaData = AudioMetaData

# Patch hyperpyyaml for Python 3.14 compatibility
# ast.Num was removed in Python 3.8+, replaced with ast.Constant
# We need to patch it before hyperpyyaml is imported
if not hasattr(ast, "Num"):
    # Add Num as an alias to Constant for backward compatibility
    class Num:
        """Backward compatibility alias for ast.Constant with numeric values"""
        pass
    
    # Monkey patch the ast module to add Num
    original_ast_eval = None
    
    def patch_hyperpyyaml():
        """Patch hyperpyyaml after it's imported"""
        try:
            import hyperpyyaml.core as hyperpyyaml_core
            global original_ast_eval
            if original_ast_eval is None:
                original_ast_eval = hyperpyyaml_core._ast_eval
            
            def patched_ast_eval(node):
                """Patched version that handles ast.Constant instead of ast.Num"""
                # Handle ast.Constant (Python 3.8+)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
                    return node.value
                # Handle ast.BinOp
                elif isinstance(node, ast.BinOp):
                    ops = {
                        ast.Add: lambda a, b: a + b,
                        ast.Sub: lambda a, b: a - b,
                        ast.Mult: lambda a, b: a * b,
                        ast.Div: lambda a, b: a / b,
                        ast.FloorDiv: lambda a, b: a // b,
                        ast.Pow: lambda a, b: a ** b,
                        ast.Mod: lambda a, b: a % b,
                    }
                    return ops[type(node.op)](patched_ast_eval(node.left), patched_ast_eval(node.right))
                # Handle ast.UnaryOp
                elif isinstance(node, ast.UnaryOp):
                    if isinstance(node.op, ast.USub):
                        return -patched_ast_eval(node.operand)
                    elif isinstance(node.op, ast.UAdd):
                        return +patched_ast_eval(node.operand)
                    else:
                        return original_ast_eval(node)
                else:
                    return original_ast_eval(node)
            
            hyperpyyaml_core._ast_eval = patched_ast_eval
        except (ImportError, AttributeError):
            pass
    
    # Patch after import
    import importlib.util
    import sys
    
    def patch_on_import(name):
        """Patch when hyperpyyaml is imported"""
        if name == 'hyperpyyaml.core':
            patch_hyperpyyaml()
    
    # Store original import
    _original_import = __import__
    
    def patched_import(name, *args, **kwargs):
        result = _original_import(name, *args, **kwargs)
        patch_on_import(name)
        return result
    
    # Apply patch
    __builtins__['__import__'] = patched_import

