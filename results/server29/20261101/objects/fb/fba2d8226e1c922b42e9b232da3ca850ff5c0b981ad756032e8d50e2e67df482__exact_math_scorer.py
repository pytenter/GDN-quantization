"""Frozen final-box extraction and exact mathematical comparison, CPU-only."""
import ast
import re
import signal
import sympy as sp
from sympy.parsing.latex import parse_latex
VERSION='HMMT26_FINAL_BOX_EXACT_V1'
FUNCTIONS={'sqrt':sp.sqrt,'cbrt':sp.cbrt,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'log':sp.log,'ln':sp.log,'exp':sp.exp,'Abs':sp.Abs}
NAMES={'pi':sp.pi,'e':sp.E,'E':sp.E,'I':sp.I,'x':sp.Symbol('x'),'y':sp.Symbol('y'),'z':sp.Symbol('z')}
def extract_final(text,thinking_expected=True):
    if thinking_expected:
        if '</think>' not in text:return dict(status='UNFINISHED_THINKING',raw=None)
        text=text.rsplit('</think>',1)[1]
    matches=list(re.finditer(r'\\(?:boxed|fbox)\s*\{',text))
    if not matches:return dict(status='NO_FINAL_BOX',raw=None)
    m=matches[-1];start=m.end();depth=1;i=start
    while i<len(text) and depth:
        if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
        elif text[i]=='}' and (i==0 or text[i-1]!='\\'):depth-=1
        i+=1
    if depth:return dict(status='INCOMPLETE_FINAL_BOX',raw=None)
    value=text[start:i-1].strip()
    return dict(status='EXTRACTED' if value else 'EMPTY_FINAL_BOX',raw=value,boxes=len(matches))
def parse_ast(text):
    def convert(node):
        if isinstance(node,ast.Expression):return convert(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in [int,float]:return sp.Rational(ast.get_source_segment(text,node))
        if isinstance(node,ast.Name) and node.id in NAMES:return NAMES[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            v=convert(node.operand);return v if isinstance(node.op,ast.UAdd) else -v
        if isinstance(node,ast.BinOp):
            a,b=convert(node.left),convert(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow):
                if b.is_Integer and abs(b)>100000:raise ValueError('Exponent outside grammar bound')
                return a**b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS and not node.keywords and len(node.args)==1:
            return FUNCTIONS[node.func.id](convert(node.args[0]))
        raise ValueError('Unsupported AST grammar')
    return convert(ast.parse(text,mode='eval'))
def parse_exact(text):
    if not isinstance(text,str) or not text.strip() or len(text)>4096:raise ValueError('Empty/oversized expression')
    text=text.strip().replace('−','-').replace(r'\dfrac',r'\frac').replace(r'\tfrac',r'\frac')
    text=re.sub(r'\^\s*(?:\{\s*\\circ\s*\}|\\circ)', '', text).replace('°','')
    if any(x in text for x in ['__','import','lambda',';','"',"'"]) or chr(96) in text:raise ValueError('Outside mathematical grammar')
    for a,b in [(r'\(',r'\)'),(r'\[',r'\]'),('$','$')]:
        if text.startswith(a) and text.endswith(b):text=text[len(a):-len(b)].strip()
    expr=parse_latex(text,strict=True) if '\\' in text or '{' in text or '}' in text else parse_ast(text.replace('^','**'))
    if not isinstance(expr,sp.Expr):raise ValueError('Expected one scalar mathematical result')
    constants={s:(sp.pi if s.name=='pi' else sp.E if s.name=='e' else sp.I if s.name in ['i','I'] else s) for s in expr.free_symbols}
    expr=expr.xreplace(constants)
    expr=expr.xreplace({v:sp.Rational(str(v)) for v in expr.atoms(sp.Float)})
    if expr.has(sp.nan,sp.zoo,sp.oo,-sp.oo):raise ValueError('Nonfinite expression')
    return expr
def _alarm(signum,frame):raise TimeoutError('Exact parser/comparison timeout')
def grade(full_output,gold,finish_reason,thinking_expected=True):
    extracted=extract_final(full_output,thinking_expected)
    result=dict(scorer=VERSION,finish_reason=finish_reason,correct=False,extraction=extracted,prediction_parsed=None,gold_parsed=None,parser_status=extracted['status'])
    if extracted['raw'] is None:return result
    previous=signal.signal(signal.SIGALRM,_alarm);signal.alarm(15)
    try:
        a=parse_exact(extracted['raw']);b=parse_exact(gold)
        result.update(prediction_parsed=sp.srepr(a),gold_parsed=sp.srepr(b))
        difference=sp.simplify(sp.together(a-b))
        if difference!=0:difference=sp.simplify(sp.radsimp(sp.trigsimp(difference)))
        if difference==0:result.update(correct=True,parser_status='EXACT_EQUAL')
        elif difference.is_zero is False:result.update(parser_status='EXACT_NOT_EQUAL')
        else:result.update(parser_status='UNRESOLVED',diagnostic='Exact symbolic simplification did not prove equality; frozen score is incorrect.')
    except Exception as e:result.update(parser_status='PARSE_FAILED' if not isinstance(e,TimeoutError) else 'PARSE_TIMEOUT',diagnostic=type(e).__name__+': '+str(e))
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    return result
