import turtle
import re
import sys
import random
import tkinter as tk
import threading
import math
from turtle import TurtleScreen, RawTurtle, TK

current_label_height = 20  # 默认标签高度
variables ={}
parameters = []
stack = []
i = 0
# Global turtle objects
turtle_obj = None
turtle_screen = None
running = True  # 全局变量，用于控制程序的运行状态

def make_tk(type, val=None):
    return [type, val]


def tk_tag(t):
    return t[0]


def tk_val(t):
    return t[1] if len(t) > 1 else None


def print_error(msg):
    print("\033[91m" + msg + "\033[0m")


def error(src, msg, exit_program=False):
    print_error(f'<{src}>: {msg}')
    if exit_program:
        sys.exit(1)


def lexer(prog):
    def err(msg):
        error('lexer', msg, True)

    pos = -1
    cur = None

    keywords = ['fd', 'bk', 'rt', 'lt', 'pu', 'pd', 'setxy', 'repeat', 'circle', 'pencolor', 'pensize',
                'hideturtle', 'showturtle', 'home', 'cs', 'label', 'if', 'while', 'make', 'to', 'end', 
                'print','random', 'for', 'stop', 'setlabelheight', 'wait']

    def next():
        nonlocal pos, cur

        t = cur

        pos += 1
        if pos >= len(prog):
            cur = 'eof'
        else:
            cur = prog[pos]

        return t

    def peek():
        return cur

    def match(m):
        if cur != m:
            err(f'Expected {m}, got {cur}')
        return next()

    def ws_skip():
        while peek() in [' ', '\t', '\r', '\n']:
            next()

    def num():
        r = next()
        has_decimal = False
        while peek() and (peek().isdigit() or (peek() == '.' and not has_decimal)):
            if peek() == '.':
                has_decimal = True
            r += next()
        return make_tk('num', float(r) if has_decimal else int(r))

    def id():
        r = next()
        while peek() and (peek().isalnum() or peek() == '_'):
            r += next()
        if r in keywords:
            return make_tk(r)
        return make_tk('id', r)

    def hexcolor():
        r = next()
        while peek() and re.match(r'[0-9a-fA-F]', peek()):
            r += next()
        if len(r) == 7:  # Expecting #RRGGBB
            return make_tk('hexcolor', r)
        err(f'Invalid hex color {r}')

    def string():
        next()  # Skip the opening quote
        r = ''
        while peek() and peek() not in ['"', ' ','\t', '\r', '\n']:
            r += next()
        if peek() in [' ','\t', '\r', '\n']:
            return make_tk('string', r)
        err(f'Invalid string literal {r}')

    def operator():
        r = next()
        if peek() in ['='] and r in ['<', '>', '=', '!']:
            r += next()
        return make_tk(r)

    def comment():
        r = ''
        while peek() and peek() != '\n':
            r += next()
        return make_tk('comment', r.strip())
    
    def quotedstring():
        next()  # Skip the opening quote
        r = ''
        while peek() and peek() != ']':
            if peek() == '[':
                # Recursively handle nested quotedstring
                nested_str = quotedstring()
                r += '['
                r += nested_str[1]  # nested_str is [type, val], we need the value part
                r += ']'
            else:
                r += next()
        if peek() == ']':
            next()  # Skip the closing quote
            return make_tk('quotedstring', r)
        else:
            err(f'Invalid quotedstring literal {r}')

    def token():
        ws_skip()
        t = peek()
        if t == 'eof':
            return make_tk('eof')
        if t in [',',']', ':', '+', '-', '*', '/']:
            next()
            return make_tk(t)
        if t in ['>', '<', '=', '!']:
            return operator()
        if t.isdigit():
            return num()
        if t.isalpha():
            return id()
        if t == '#':
            return hexcolor()
        if t == '"':
            return string()
        if t == ';':  # Assuming ';' starts a comment
            return comment()
        if tokens[-1][0] == 'print' and t == '[':
            return quotedstring()
        elif t == '[':
            next()
            return make_tk(t)

        err(f'Illegal character {t}')

    # lexer start
    next()

    tokens = []

    while True:
        t = token()
        if tk_tag(t) == 'eof':
            break
        tokens.append(t)

    return tokens


def make_tokenizer(tokens, err):
    cur = None
    pos = -1

    def next():
        nonlocal cur, pos

        t = cur

        pos += 1
        if pos >= len(tokens):
            cur = ['eof']
        else:
            cur = tokens[pos]

        return t

    def peek(k=0):
        if k + pos < len(tokens):
            return tk_tag(tokens[pos + k])
        else:
            return 'eof'

    def match(*m):
        if peek() not in m:
            err(f'Expected {m}, got {peek()}')
        return next()

    next()

    return (next, peek, match)


def logo_parser(tokens):
    def err(m):
        error('logo parser', m)

    next, peek, match = make_tokenizer(tokens, err)

    def command():
        tok = peek()
        if tok == 'fd':
            return forward()
        elif tok == 'print':
            return print_command()
        elif tok == 'id':
            name = tk_val(next())
            params = []
            while True:
                # if peek() in ['eof','end']:
                if peek() in ':':
                    params.append(expression())
                elif peek() in 'num':
                    params.append(expression())
                if peek() ==',':
                    next()
                else:
                    break
            return {'cmd': 'call', 'name': name,'params':params}
        elif tok == 'bk':
            return backward()
        elif tok == 'rt':
            return right()
        elif tok == 'lt':
            return left()
        elif tok == 'pu':
            return penup()
        elif tok == 'pd':
            return pendown()
        elif tok == 'setxy':
            return setxy()
        elif tok == 'repeat':
            return repeat()
        elif tok == 'circle':
            return circle()
        elif tok == 'pencolor':
            return pencolor()
        elif tok == 'pensize':
            return pensize()
        elif tok == 'hideturtle':
            return hideturtle()
        elif tok == 'showturtle':
            return showturtle()
        elif tok == 'home':
            return home()
        elif tok == 'cs':
            return cs()
        elif tok == 'label':
            return label()
        elif tok == 'setlabelheight':
            return setlabelheight()
        elif tok == 'wait': 
            return wait()
        elif tok == 'make':
            return make()
        elif tok == 'if':
            return if_statement()
        elif tok == 'while':
            return while_loop()
        elif tok == 'to':
            return procedure_definition()
        elif tok == 'end':
            return end_statement()
        elif tok == 'comment':
            return comment()
        elif tok == 'random':
            return random_num()
        elif tok == 'for':
            return for_loop()
        elif tok == 'stop':
            return stop()
        else:
            err('Unexpected command')

    def stop():
        next()
        return {'cmd':'stop'}
    def random_num():
        next()
        num = match('num')[1]
        return  {'type':'random','num':num}
    def forward():
        match('fd')
        if peek() == 'random':
            dist = command()
        else :
            dist = expression()
        return {'cmd': 'fd', 'dist': dist}

    def backward():
        match('bk')
        dist = expression()
        return {'cmd': 'bk', 'dist': dist}

    def right():
        match('rt')
        angle = expression()
        return {'cmd': 'rt', 'angle': angle}

    def left():
        match('lt')
        angle = expression()
        return {'cmd': 'lt', 'angle': angle}
    
    def print_command():
        match('print')
        value = expression()
        return {'cmd': 'print', 'value': value}

    def penup():
        match('pu')
        return {'cmd': 'pu'}

    def pendown():
        match('pd')
        return {'cmd': 'pd'}

    def setxy():
        match('setxy')
        x = expression()
        y = expression()
        return {'cmd': 'setxy', 'x': x, 'y': y}

    def repeat():
        match('repeat')
        count = expression()
        match('[')
        cmds = []
        while peek() != ']':
            cmds.append(command())
        match(']')
        return {'cmd': 'repeat', 'count': count, 'cmds': cmds}

    def circle():
        match('circle')
        radius = expression()
        return {'cmd': 'circle', 'radius': radius}

    def pencolor():
        match('pencolor')
        if peek() == 'id':
            color = match('id')
            return {'cmd': 'pencolor', 'color': color[1]}
        elif peek() == 'hexcolor':
            color = match('hexcolor')
            return {'cmd': 'pencolor', 'color': color[1]}
        else:
            r = match('num')
            g = match('num')
            b = match('num')
            return {'cmd': 'pencolor', 'color': (r[1], g[1], b[1])}

    def pensize():
        match('pensize')
        size = expression()
        return {'cmd': 'pensize', 'size': size}

    def hideturtle():
        match('hideturtle')
        return {'cmd': 'hideturtle'}

    def showturtle():
        match('showturtle')
        return {'cmd': 'showturtle'}

    def home():
        match('home')
        return {'cmd': 'home'}

    def cs():
        match('cs')
        return {'cmd': 'cs'}

    def label():
        match('label')
        text = tk_val(match('string'))
        return {'cmd': 'label', 'text': text}
    
    def setlabelheight():
        match('setlabelheight')
        height = expression()
        return {'cmd': 'setlabelheight', 'height': height}

    def wait():
        match('wait')
        duration = expression()
        return {'cmd': 'wait', 'duration': duration}

    def make():
        match('make')
        var_name = match('string')[1]
        value = expression()
        return {'cmd': 'make', 'var': var_name, 'value': value}

    def if_statement():
        match('if')
        condition = comparison()
        match('[')
        cmds = []
        while peek() != ']':
            cmds.append(command())
        match(']')
        return {'cmd': 'if', 'condition': condition, 'cmds': cmds}

    def while_loop():
        match('while')
        condition = comparison()
        match('[')
        cmds = []
        while peek() != ']':
            cmds.append(command())
        match(']')
        return {'cmd': 'while', 'condition': condition, 'cmds': cmds}

    def procedure_definition():
        match('to')
        name = match('id')[1]
        params = []
        while True:
            if peek() == ':':
                params.append(expression())
            if peek() != ',':
                break
            else: next()
        cmds = []
        while peek() != 'end':
            cmds.append(command())
        match('end')
        return {'cmd': 'procedure', 'name': name, 'params': params, 'cmds': cmds}

    def end_statement():
        match('end')
        return {'cmd': 'end'}

    def expression():
        def term():
            token = next()
            if tk_tag(token) == 'num':
                return {'type': 'num', 'value': tk_val(token)}
            elif tk_tag(token) == 'id':
                return {'type': 'var', 'name': tk_val(token)}
            elif tk_tag(token) == ':':
                var_name = match('id')[1]
                return {'type': 'var', 'name': var_name}
            elif tk_tag(token) == 'string':
                return {'type': 'string', 'value': tk_val(token)}
            elif tk_tag(token) == 'quotedstring':
                return {'type': 'quotedstring', 'value': tk_val(token)}
            else:
                err('Invalid term')

        def factor():
            t = term()
            while peek() in ['*', '/']:
                op = next()
                t = {'type': 'binop', 'op': tk_tag(op), 'left': t, 'right': term()}
            return t

        expr = factor()
        while peek() in ['+', '-']:
            op = next()
            expr = {'type': 'binop', 'op': tk_tag(op), 'left': expr, 'right': factor()}
        return expr

    def comparison():
        left = expression()
        op = match('>', '<', '==', '>=', '<=', '!=')
        right = expression()
        return {'left': left, 'op': tk_tag(op), 'right': right}

    def comment():
        cmt = match('comment')
        return {'cmd': 'comment', 'cmt': cmt[1]}
    
    def for_loop():
        match('for')
        match('[')
        var_name = tk_val(match('id'))
        start_expr = expression()
        end_expr = expression()
        step_expr = expression()
        match(']')
        match('[')
        cmds = []
        while peek() != ']':
            cmds.append(command())
        match(']')
        return {'cmd': 'for', 'var': var_name, 'start': start_expr, 'end': end_expr, 'step': step_expr, 'cmds': cmds}

    def program():
        cmds = []
        while peek() != 'eof':
            cmds.append(command())
        return {'program': cmds}

    return program()


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

class StopExecution(Exception):
    pass

def evaluate_expression(expr):
    global variables
    if expr['type'] == 'num':
        value_str = str(expr['value'])
        if '.' in value_str :
            parts = value_str.split('.')
            if parts[1]== '0':
                return int(parts[0])
            else:
                return float(value_str)
        else:
            return int(value_str)
    elif expr['type'] == 'var':
        value_str = str(variables[expr['name']])
        if '.' in value_str :
            parts = value_str.split('.')
            if parts[1]== '0':
                return int(parts[0])
            else:
                return float(value_str)
        else:
            return int(value_str)
    elif expr['type'] == 'string':
        return expr['value']
    elif expr['type'] == 'binop':
        left = evaluate_expression(expr['left'])
        right = evaluate_expression(expr['right'])
        if expr['op'] == '+':
            return left + right
        elif expr['op'] == '-':
            return left - right
        elif expr['op'] == '*':
            return left * right
        elif expr['op'] == '/':
            return left / right
        else:
            raise ValueError("Unknown operator")
    elif expr['type'] == 'random':
        num = int(expr['num'])
        return random.randint(0,num)
    if expr['type'] == 'quotedstring':
        return expr['value']  # 返回字符串的值
    else:
        raise ValueError("Unknown expression type")

def execute_command(cmd, t, screen):
    global current_label_height  # 声明全局变量
    global running
    global variables
    if not running:
        return  # 如果停止按钮被按下，直接返回
    if cmd['cmd'] == 'fd':
        t.forward(evaluate_expression(cmd['dist']))
    elif cmd['cmd'] == 'bk':
        t.backward(evaluate_expression(cmd['dist']))
    elif cmd['cmd'] == 'rt':
        t.right(evaluate_expression(cmd['angle']))
    elif cmd['cmd'] == 'lt':
        t.left(evaluate_expression(cmd['angle']))
    elif cmd['cmd'] == 'pu':
        t.penup()
    elif cmd['cmd'] == 'pd':
        t.pendown()
    elif cmd['cmd'] == 'setxy':
        t.setx(evaluate_expression(cmd['x']))
        t.sety(evaluate_expression(cmd['y']))
    elif cmd['cmd'] == 'circle':
        t.circle(evaluate_expression(cmd['radius']))
    elif cmd['cmd'] == 'pencolor':
        color = cmd['color']
        if isinstance(color, tuple):
            color = rgb_to_hex(color)
        t.pencolor(color)
    elif cmd['cmd'] == 'pensize':
        t.pensize(evaluate_expression(cmd['size']))
    elif cmd['cmd'] == 'hideturtle':
        t.hideturtle()
    elif cmd['cmd'] == 'showturtle':
        t.showturtle()
    elif cmd['cmd'] == 'repeat':
        count = evaluate_expression(cmd['count'])
        try:
            for _ in range(count):
                for sub_cmd in cmd['cmds']:
                    execute_command(sub_cmd, t, screen)
        except StopExecution:
            pass  # 捕获到停止异常后，直接退出循环
    elif cmd['cmd'] == 'home':
        t.penup()
        t.home()
        t.pendown()
    elif cmd['cmd'] == 'cs':
        t.clear()
        t.penup()
        t.home()
        t.pendown()
    elif cmd['cmd'] == 'label':
        value = cmd['text']
        value = str(value)
        len_v = len(value)
        angle = t.heading()  # 获取当前turtle的方向
        start_x, start_y = t.pos()
        angle_rad = math.radians(angle)
        # 调整文本起始位置
        offset_x = 0# 没有偏移
        offset_y = -current_label_height / len_v 
        x_new = start_x+offset_x * math.cos(angle_rad) - offset_y * math.sin(angle_rad)
        y_new = start_y+offset_x * math.sin(angle_rad) + offset_y * math.cos(angle_rad)

        t.pu()
        t.ht()
        t.goto(x_new, y_new)
        t.pd()
       
        canvas.create_text(x_new, y_new, text=value, angle=-angle, font=("Arial", current_label_height, "normal"), anchor=tk.NW)
        t.pu()
        t.fd(current_label_height)
        t.pd()
        t.pu()
        t.goto(x_new, y_new)
            
    elif cmd['cmd'] == 'setlabelheight':
        current_label_height = evaluate_expression(cmd['height'])
    elif cmd['cmd'] == 'wait':
        duration = evaluate_expression(cmd['duration'])
        # screen.ontimer(None, t=duration)
        canvas.after(int(duration*1000))
    elif cmd['cmd'] == 'procedure':
        variables[cmd['name']] = cmd['cmds']
        for p in cmd['params']:
            # parameters[p['name']] = None
            variables[p['name']] = None
            parameters.append(p['name'])
        # for to_cmd in cmd['cmds']:
        #     execute_command(to_cmd,t,screen)
    elif cmd['cmd'] == 'make':
        variables[cmd['var']] = evaluate_expression(cmd['value'])
    elif cmd['cmd'] == 'if':
        condition = cmd['condition']
        left = evaluate_expression(condition['left'])
        right = evaluate_expression(condition['right'])
        op = condition['op']
        if ((op == '>' and left > right) or
                (op == '<' and left < right) or
                (op == '==' and left == right) or
                (op == '>=' and left >= right) or
                (op == '<=' and left <= right) or
                (op == '!=' and left != right)):
            for sub_cmd in cmd['cmds']:
                execute_command(sub_cmd, t, screen)
    elif cmd['cmd'] == 'while':
        condition = cmd['condition']
        while True:
            left = evaluate_expression(condition['left'])
            right = evaluate_expression(condition['right'])
            op = condition['op']
            if not ((op == '>' and left > right) or
                    (op == '<' and left < right) or
                    (op == '==' and left == right) or
                    (op == '>=' and left >= right) or
                    (op == '<=' and left <= right) or
                    (op == '!=' and left != right)):
                break
            try:
                for sub_cmd in cmd['cmds']:
                    execute_command(sub_cmd, t, screen)
            except StopExecution:
                break  # 捕获到停止异常后，直接退出循环
    elif cmd['cmd'] == 'comment':
        pass  # Ignore comments
    elif cmd['cmd'] == 'call':
        call_name = cmd['name']
        call_cmd = variables[call_name]
        call_params = []
        for i in cmd['params']:
            call_params.append(evaluate_expression(i))
        stack.append(variables.copy())

        for i in range(len(call_params)):
            variables[parameters[i]] = call_params[i]
        try:
            for cmd in call_cmd:
                # print(variables['size'])
                execute_command(cmd, t, screen)
        except StopExecution:
            variables = stack.pop()


    elif cmd['cmd'] == 'print':
        value = evaluate_expression(cmd['value'])
        if isinstance(value, int):
            value = int(value)
        if isinstance(value,float):
            value = float(value)
        value = str(value)
        size = 10
        t.pu()
        t.ht()
        t.fd(0)
        t.pd()
        
        for i in value:
            t.write(i,font=('Times New Roman',size,'normal'))
            t.pu()
            t.fd(size)
            t.pd()
    elif cmd['cmd'] == 'for':
        var_name = cmd['var']
        start_val = evaluate_expression(cmd['start'])
        end_val = evaluate_expression(cmd['end'])
        step_val = evaluate_expression(cmd['step'])
        variables[var_name] = start_val
        try:
            while variables[var_name] <= end_val:
                for sub_cmd in cmd['cmds']:
                    execute_command(sub_cmd, t, screen)
                variables[var_name] += step_val
        except StopExecution:
            pass  # 捕获到停止异常后，直接退出循环
    elif cmd['cmd'] == 'stop':
        raise StopExecution

    else:
        raise ValueError(f"Unknown command {cmd['cmd']}")

def execute_program(prog):
    global turtle_obj
    for cmd in prog['program']:
        if not running:
            break  # 如果停止按钮被按下，退出循环
        execute_command(cmd, turtle_obj, canvas)

def run_logo(prog):
    global turtle_obj
    global turtle_screen

    # 清屏操作
    turtle_obj.reset()
    canvas.delete("all")
    turtle_obj.clear()
    turtle_obj.penup()
    turtle_obj.home()
    turtle_obj.pendown()
    turtle_screen.update()
    turtle_screen.clearscreen


    tokens = lexer(prog)
    ast = logo_parser(tokens)
    execute_program(ast)


def stop_execution():
    global running
    running = False

def run_code():
    global running
    running = True
    code = code_text.get("1.0", tk.END)
    threading.Thread(target=run_logo, args=(code,)).start()

def clear_code():
    code_text.delete("1.0", tk.END)

def insert_code_to_editor(code):
    code_text.delete("1.0", tk.END)
    code_text.insert(tk.END, code.strip())
    highlight_code(code_text)

def highlight_code(text_widget):
   # Clear previous tags
    text_widget.tag_configure("variable_highlight", foreground="#EE3F11")  # Configure tag for variables
    text_widget.tag_configure("to_end_highlight", foreground="#c53ac2") 

    text_content = text_widget.get("1.0", tk.END)

    var_pattern = r'(.:[a-zA-Z]+| "[a-zA-Z]+)'
    for match in re.finditer(var_pattern, text_content):
        start = text_widget.index(f"1.0 + {match.start()} chars")
        end = text_widget.index(f"1.0 + {match.end()} chars")
        text_widget.tag_add("variable_highlight", start, end)
    to_end = r'\b(to|end|stop|for|repeat|if)'
    for match in re.finditer(to_end, text_content):
        start = text_widget.index(f"1.0 + {match.start()} chars")
        end = text_widget.index(f"1.0 + {match.end()} chars")
        text_widget.tag_add("to_end_highlight", start, end)


def insert_code_from_label(event, index, code):
    insert_code_to_editor(code)

root = tk.Tk()
root.title("Logo Interpreter")

left_frame = tk.Frame(root, padx=5, pady=10)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

top_frame = tk.Frame(left_frame)
top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
tk.Label(top_frame, text="Logo Interpreter", font=("Comic Sans MS", 24), anchor="w", padx=20).pack(side=tk.LEFT, fill=tk.X)

mid_frame = tk.Frame(left_frame)
mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas = tk.Canvas(mid_frame, width=600, height=400, bg="white")
canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
turtle_screen = turtle.TurtleScreen(canvas)
turtle_screen.screensize(600,400)
turtle_obj = turtle.RawTurtle(turtle_screen)

bottom_frame = tk.Frame(left_frame)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
code_text = tk.Text(bottom_frame, wrap=tk.WORD, height=10)
code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
scrollbar = tk.Scrollbar(bottom_frame, command=code_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
code_text.config(yscrollcommand=scrollbar.set)

button_frame = tk.Frame(bottom_frame)
button_frame.pack(side=tk.RIGHT, fill=tk.Y)
run_button = tk.Button(button_frame, text="执行", command=run_code, width=10)
run_button.pack(side=tk.TOP, pady=10)
stop_button = tk.Button(button_frame, text="停止", command=stop_execution, width=10)
stop_button.pack(side=tk.TOP, pady=10)
clear_button = tk.Button(button_frame, text="清除", command=clear_code, width=10)
clear_button.pack(side=tk.BOTTOM, pady=10)

right_frame = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
tk.Label(right_frame, text="参考代码", font=("黑体", 18)).pack(anchor=tk.W)

example_codes = [
    """
pencolor red
make "x 30
if :x > 5 [
    repeat 5 [ fd :x rt 144 ]
    cs
    pencolor #FF00FF
    circle :x
    cs
    pencolor 0 0 144
    setxy 30 30
    bk :x
]
    """,
    """
make "x 1
repeat 5 [
	repeat 5 [
	if :x > 3 [stop]
	fd :x * 10
	print :x
	make "x :x + 1
	]
	rt 90
	fd x * 10
	make "x :x + 1
]
pencolor red
pensize 4
circle :x
""",
"""
hideturtle  
make "x 1
repeat 100 [

    setlabelheight :x
    pu
    fd :x * :x / 15
    label "LOGO
    pu
    bk :x * :x / 15
    rt 10
    make "x :x+1
]
showturtle
"""
,
"""
print [hello [world]]
repeat 5 [fd random 80 rt 90]
make "x 1
for [i :x 10 2] [

    for [j i+1 10 2][
    if :j >7 [stop]
    print :j
    fd 10
]
fd 40
rt 90
]
"""
]

def insert_example_text(example_text, index):
    text_widget = tk.Text(right_frame, wrap=tk.WORD, height=10, width=40, font=("Times New Roman", 11), padx=5, pady=5, cursor="hand2", bg="#D3D3D3")
    text_widget.insert(tk.END, example_text.strip())
    text_widget.pack(fill=tk.X, padx=5, pady=(0,5))
    text_widget.bind("<Button-1>", lambda e, index=index, code=example_text: insert_code_from_label(e, index, code))
    highlight_code(text_widget)  # Highlight initially based on content
    text_widget.config(state='disable')

for i, example in enumerate(example_codes):
    insert_example_text(example, i)

root.mainloop()