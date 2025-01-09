from pyautocad import Autocad, APoint

# Initialize AutoCAD
acad = Autocad(create_if_not_exists=True)
print(f"Connected to AutoCAD: {acad.app.Name}")

# Example: Adding a Circle
def add_circle(center_x, center_y, radius):
    center = APoint(center_x, center_y)
    acad.model.AddCircle(center, radius)
    print(f"Circle created at ({center_x}, {center_y}) with radius {radius}")

# Example: Adding a Line
def add_line(start_x, start_y, end_x, end_y):
    start = APoint(start_x, start_y)
    end = APoint(end_x, end_y)
    acad.model.AddLine(start, end)
    print(f"Line created from ({start_x}, {start_y}) to ({end_x}, {end_y})")

# Test the functions
add_circle(100, 100, 50)
add_line(0, 0, 200, 200)