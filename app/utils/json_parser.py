import json


def points_to_bbox(points: list[dict]) -> dict:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    return {
        "x": float(min_x),
        "y": float(min_y),
        "w": float(max_x - min_x),
        "h": float(max_y - min_y),
    }


def parse_bounding_boxes_json(raw_bytes: bytes) -> list[dict]:
    data = json.loads(raw_bytes.decode("utf-8"))

    if not isinstance(data, list):
        raise ValueError("bounding_boxes.json must be a list")

    parsed = []

    for i, item in enumerate(data):
        if "points" not in item:
            raise ValueError(f"Missing 'points' in item index {i}")

        raw_points = item["points"]

        if not isinstance(raw_points, list) or len(raw_points) < 4:
            raise ValueError(f"'points' must contain at least 4 coordinate pairs in item index {i}")

        points = []
        for j, pt in enumerate(raw_points):
            if not isinstance(pt, list) or len(pt) != 2:
                raise ValueError(f"Invalid point format at item {i}, point {j}")

            x, y = pt
            points.append({
                "x": float(x),
                "y": float(y),
            })

        bbox = points_to_bbox(points)

        parsed.append({
            "label": str(item.get("label", f"slot_{i + 1:03d}")),
            "points": points,
            "bbox": bbox,
        })

    return parsed