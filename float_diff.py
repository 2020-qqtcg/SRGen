def calculate_difference(a: float, b: float) -> float:
    """
    计算两个浮点数的差值
    
    Args:
        a (float): 第一个浮点数
        b (float): 第二个浮点数
        
    Returns:
        float: a和b的差值 (a - b)
    """
    return a - b

# 示例使用
if __name__ == "__main__":
    num1 = 1757879517.9562836
    num2 = 1757877699.0292687
    result = calculate_difference(num1, num2)
    result = result / 30
    print(f"{num1} - {num2} = {result}")