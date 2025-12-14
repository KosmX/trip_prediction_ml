import http.server
import json
import socketserver

import _socket
import pandas as pd

from baseline import BaselineModel
from mlp import MLPModel
from model import Model
from data import DataLoader

with open('/app/data/routes.json', 'r') as f:
    routes_data = json.load(f)


# Datetime is missing, but I do not want to fix, because without types, that would take at least 2 hours
# please, next time use any language with types :(
def generate_timetable(model: Model, route_name: str, direction: bool, start_time='08:00:00'):
    selected_route = None
    for route in routes_data:
        if route['name'] == route_name and route['direction'] == direction:
            selected_route = route
            break

    if selected_route is None:
        print(f"Error: Route '{route_name}' not found.")
        return None

    stops = selected_route['stops']

    # Initialize timetable
    timetable = []
    current_time_str = start_time
    # Convert start_time to datetime object for calculations
    current_time = pd.to_datetime(start_time, format='%H:%M:%S')

    # Add the first stop with its initial arrival time
    timetable.append({
        'stop': stops[0],
        'arrival_time': current_time.strftime('%H:%M:%S')
    })

    # Iterate through predicted durations to build the timetable
    for i in range(1, len(stops)):
        prev = stops[i - 1]
        curr = stops[i]

        duration_seconds = model.predict_on(prev, curr, str(current_time), selected_route['route_type'])['predicted_duration']
        # Add duration to current_time
        current_time += pd.to_timedelta(duration_seconds, unit='s')

        timetable.append({
            'stop': stops[i],
            'arrival_time': current_time.strftime('%H:%M:%S')
        })

    # Convert timetable to DataFrame for better readability and return
    timetable_df = pd.DataFrame(timetable)
    return timetable_df


if __name__ == '__main__':
    data = DataLoader('/app/data').all()
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    X_test = test.drop(columns=['duration'])

    baseline = BaselineModel()
    baseline.train(train, test)

    test_predictions = baseline.predict(X_test)
    # evaluate with RMSE
    rmse = ((test['duration'] - test_predictions['predicted_duration']) ** 2).mean() ** 0.5
    print(f"Baseline RMSE: {rmse}")

    print("------------------------")
    print("Test timetable of 56 route (with baseline):")
    timetable_56 = generate_timetable(baseline, '56', True, '08:00:00')
    print(timetable_56)
    print("------------------------")

    mlp = MLPModel()
    mlp.train(train, test)
    print("MLP model trained.")
    test_predictions_mlp = mlp.predict(X_test)
    rmse = ((test['duration'] - test_predictions_mlp['predicted_duration']) ** 2).mean() ** 0.5
    print(f"MLP RMSE: {rmse}")


    print("------------------------")
    print("Test timetable of 56 route (with MLP):")
    timetable_56 = generate_timetable(mlp, '56', True, '08:00:00')
    print(timetable_56)
    print("------------------------")


    models: list[Model] = [mlp, baseline]

    print("All good :), starting server...")

    class Server(http.server.SimpleHTTPRequestHandler):

        def do_GET(self):
            if self.path.startswith('/timetable'):
                # Parse query parameters
                from urllib.parse import urlparse, parse_qs
                query_components = parse_qs(urlparse(self.path).query)
                route_name = query_components.get('route', [None])[0]
                direction = query_components.get('direction', ['0'])[0] == '1'
                start_time = query_components.get('start_time', ['08:00:00'])[0]

                if route_name is None:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'Missing route parameter')
                    return

                timetable = generate_timetable(models[0], route_name, direction, start_time)

                if timetable is None:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Route not found')
                    return

                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(timetable.to_json(orient='records').encode())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')


    server_address = ('', 8080)
    with socketserver.TCPServer(server_address, Server) as httpd:
        print("Starting server on port 8080...")
        httpd.serve_forever()
