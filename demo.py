import tkinter as tk
from tkinter import ttk, scrolledtext, Spinbox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import shap

# Create more realistic sample maintenance data with different conditions
maintenance_data = {
    'Feature Name': ['oil_pressure', 'oil_pressure', 'oil_pressure',
                    'coolant_temp', 'coolant_temp', 'coolant_temp',
                    'engine_rpm', 'engine_rpm', 'engine_rpm',
                    'fuel_pressure', 'fuel_pressure', 'fuel_pressure',
                    'coolant_pressure', 'coolant_pressure', 'coolant_pressure',
                    'lub_oil_temp', 'lub_oil_temp', 'lub_oil_temp'],
    'Condition': ['Poor', 'Moderate', 'Good',
                 'Poor', 'Moderate', 'Good',
                 'Poor', 'Moderate', 'Good',
                 'Poor', 'Moderate', 'Good',
                 'Poor', 'Moderate', 'Good',
                 'Poor', 'Moderate', 'Good'],
    'Issue Description': [
        'Critically low oil pressure - immediate attention required', 'Oil pressure below optimal range', 'Oil pressure normal',
        'Dangerously high coolant temperature', 'Coolant temperature slightly elevated', 'Coolant temperature normal',
        'Severe RPM fluctuations detected', 'Minor RPM irregularities', 'RPM within normal range',
        'Critical fuel pressure drop', 'Fuel pressure slightly low', 'Fuel pressure normal',
        'Significant coolant pressure loss', 'Minor coolant pressure deviation', 'Coolant pressure normal',
        'Oil temperature critically high', 'Oil temperature slightly elevated', 'Oil temperature normal'
    ],
    'Maintenance Recommendation': [
        'Immediate inspection of oil pump and pressure relief valve', 'Monitor oil pressure and schedule maintenance', 'Continue regular maintenance',
        'Emergency cooling system inspection required', 'Check coolant levels and fan operation', 'Continue monitoring',
        'Urgent engine timing and fuel system check', 'Verify fuel quality and injector performance', 'Maintain current service schedule',
        'Emergency fuel pump and line inspection', 'Check fuel filter and pressure regulator', 'Continue regular fuel system maintenance',
        'Immediate cooling system leak inspection', 'Inspect pressure cap and hoses', 'Continue regular maintenance',
        'Urgent oil cooler and lubrication check', 'Monitor oil level and cooling efficiency', 'Continue regular oil changes'
    ]
}

df_maintenance = pd.DataFrame(maintenance_data)

class EngineDiagnosticSystem:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.model = None
        self.explainer = None
        self.feature_thresholds = {
            'oil_pressure': {'Poor': 2, 'Moderate': 4},
            'coolant_temp': {'Poor': 180, 'Moderate': 160},
            'engine_rpm': {'Poor': 2000, 'Moderate': 1800},
            'fuel_pressure': {'Poor': 5, 'Moderate': 10},
            'coolant_pressure': {'Poor': 2, 'Moderate': 4},
            'lub_oil_temp': {'Poor': 85, 'Moderate': 80}
        }

    def train_model(self, X_train, y_train):
        self.model = GradientBoostingClassifier(
            n_estimators=100,  # Increased from 50
            max_depth=4,       # Increased from 3
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.model)  # Changed to TreeExplainer

    def get_feature_condition(self, value, feature_name):
        """
        Determine feature condition based on predefined thresholds
        """
        thresholds = self.feature_thresholds[feature_name]
        
        if feature_name in ['coolant_temp', 'engine_rpm', 'lub_oil_temp']:
            if value >= thresholds['Poor']:
                return 'Poor'
            elif value >= thresholds['Moderate']:
                return 'Moderate'
            return 'Good'
        else:
            if value <= thresholds['Poor']:
                return 'Poor'
            elif value <= thresholds['Moderate']:
                return 'Moderate'
            return 'Good'

    def analyze_features(self, input_features):
        """
        Analyze features using both threshold-based and model-based approaches
        """
        input_array = np.array(input_features).reshape(1, -1)
        shap_values = self.explainer.shap_values(input_array)
        
        if isinstance(shap_values, list):  # Handle multi-class case
            shap_values = shap_values[1]  # Assuming binary classification, use class 1

        analysis = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': input_features,
            'SHAP Value': shap_values[0]
        })

        analysis['Impact'] = analysis['SHAP Value'].abs()
        analysis = analysis.sort_values(by='Impact', ascending=False)

        report = []
        concerning_features = []

        for _, row in analysis.iterrows():
            feature_name = row['Feature']
            value = row['Value']
            condition = self.get_feature_condition(value, feature_name)

            maintenance_info = df_maintenance[
                (df_maintenance['Feature Name'] == feature_name) & 
                (df_maintenance['Condition'] == condition)
            ]

            if not maintenance_info.empty:
                description = maintenance_info['Issue Description'].iloc[0]
                recommendation = maintenance_info['Maintenance Recommendation'].iloc[0]
            else:
                description = f"Unusual {feature_name} reading detected"
                recommendation = f"Please inspect {feature_name} system"

            feature_report = {
                'Feature': feature_name,
                'Current Value': value,
                'Condition': condition,
                'Description': description,
                'Recommendation': recommendation,
                'SHAP Impact': abs(row['SHAP Value'])
            }
            report.append(feature_report)

            if condition in ['Poor', 'Moderate']:
                concerning_features.append(feature_report)

        return {
            'overall_status': 'Critical' if any(f['Condition'] == 'Poor' for f in concerning_features)
                            else 'Warning' if concerning_features 
                            else 'Good',
            'all_features': report,
            'concerning_features': concerning_features
        }

class DiagnosticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Engine Diagnostic System")
        self.root.geometry("1000x800")

        self.feature_names = ['oil_pressure', 'coolant_temp', 'engine_rpm',
                            'fuel_pressure', 'coolant_pressure', 'lub_oil_temp']
        self.diagnostic_system = EngineDiagnosticSystem(self.feature_names)
        
        self.initialize_model()
        self.create_gui()

    def initialize_model(self):
        try:
            # Generate more realistic sample data if file not found
            data = pd.read_csv(r'C:\Users\RITHUN\Desktop\project\engine_data .csv')
            X = data.drop(columns=['Engine Condition'])
            y = data['Engine Condition']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.diagnostic_system.train_model(X_train, y_train)
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def create_gui(self):
        style = ttk.Style()
        style.configure('Critical.TLabel', foreground='red', font=('Helvetica', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='orange', font=('Helvetica', 10, 'bold'))
        style.configure('Good.TLabel', foreground='green', font=('Helvetica', 10, 'bold'))

        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Engine Parameters", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Output Frame
        output_frame = ttk.LabelFrame(self.root, text="Diagnostic Results", padding="10")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.inputs = {}
        
        spinbox_configs = {
            'oil_pressure': (0, 8, "Oil Pressure (PSI)", 0.1),
            'coolant_temp': (60, 200, "Coolant Temperature (°F)", 1),
            'engine_rpm': (0, 2500, "Engine RPM", 50),
            'fuel_pressure': (0, 22, "Fuel Pressure (PSI)", 0.1),
            'coolant_pressure': (0, 8, "Coolant Pressure (PSI)", 0.1),
            'lub_oil_temp': (70, 90, "Lubrication Oil Temperature (°F)", 0.1)
        }

        for feature, (min_val, max_val, label, increment) in spinbox_configs.items():
            frame = ttk.Frame(input_frame)
            frame.pack(fill="x", pady=2)
            
            ttk.Label(frame, text=label, width=30).pack(side="left")
            
            spinbox = Spinbox(
                frame,
                from_=min_val,
                to=max_val,
                increment=increment,
                width=15
            )
            spinbox.pack(side="left")
            spinbox.delete(0, tk.END)
            spinbox.insert(0, (max_val + min_val) / 2)  # Set default value
            self.inputs[feature] = spinbox

        ttk.Button(
            input_frame, 
            text="Analyze Engine", 
            command=self.analyze_engine,
            style='Accent.TButton'
        ).pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=20, wrap=tk.WORD)
        self.output_text.pack(fill="both", expand=True)

        # Configure text tags
        self.output_text.tag_configure("critical", foreground="red", font=("Helvetica", 10, "bold"))
        self.output_text.tag_configure("warning", foreground="orange", font=("Helvetica", 10, "bold"))
        self.output_text.tag_configure("good", foreground="green", font=("Helvetica", 10, "bold"))
        self.output_text.tag_configure("header", font=("Helvetica", 11, "bold"))

    def analyze_engine(self):
        try:
            input_values = [float(self.inputs[feature].get()) for feature in self.feature_names]
            diagnosis = self.diagnostic_system.analyze_features(input_values)

            self.output_text.delete(1.0, tk.END)

            # Display overall status with appropriate styling
            status_tag = diagnosis['overall_status'].lower()
            self.output_text.insert(tk.END, "Overall Engine Status: ", "header")
            self.output_text.insert(tk.END, f"{diagnosis['overall_status']}\n\n", status_tag)

            if diagnosis['concerning_features']:
                self.output_text.insert(tk.END, "Features Requiring Attention:\n\n", "header")
                for issue in sorted(diagnosis['concerning_features'], 
                                  key=lambda x: (x['Condition'] == 'Poor', x['SHAP Impact']), 
                                  reverse=True):
                    condition_tag = "critical" if issue['Condition'] == "Poor" else "warning"
                    
                    self.output_text.insert(tk.END, f"• {issue['Feature']}\n", condition_tag)
                    self.output_text.insert(tk.END, f"  Current Value: {issue['Current Value']:.2f}\n")
                    self.output_text.insert(tk.END, f"  Condition: {issue['Condition']}\n")
                    self.output_text.insert(tk.END, f"  Issue: {issue['Description']}\n")
                    self.output_text.insert(tk.END, f"  Recommendation: {issue['Recommendation']}\n")
                    self.output_text.insert(tk.END, f"  Impact Level: {issue['SHAP Impact']:.4f}\n\n")
            else:
                self.output_text.insert(tk.END, "All features are within normal operating ranges.\n", "good")

        except Exception as e:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Error during analysis: {str(e)}\n", "critical")

def main():
    root = tk.Tk()
    app = DiagnosticGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()