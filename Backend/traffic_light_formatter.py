class TrafficLightFormatter:
    def __init__(self):
        self.type_map = {
            "vehicle": "svetofor",
            "pedestrian": "piyodalar svetofori",
        }
        self.state_map = {
            "red": "qizil",
            "yellow": "sariq",
            "green": "yashil",
            "walk": "yuring",
            "dont_walk": "to'xtang",
        }
        self.zone_map = {
            "center": "Oldinda",
            "left": "Chapda",
            "right": "O'ngda",
        }

    def format_phrase(self, tl_type, state, zone, distance, size_class="medium"):
        human_type = self.type_map.get(tl_type, "svetofor")
        zone_str = self.zone_map.get(zone, "Oldinda")
        dist_str = ", yaqin" if distance == "near" else ""

        color_str = ""
        if state != "unknown" and size_class != "small":
            color_label = self.state_map.get(state)
            if color_label:
                color_str = f", {color_label}"

        return f"{zone_str} {human_type}{color_str}{dist_str}".strip()
