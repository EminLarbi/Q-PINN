Repositorio con implementaciones de PINNs para distintas configuraciones de potencial cuántico.

## Estructura del proyecto

- **gaussian_wavepacket/**  
  - `model.py`  
  - `utils.py`  
  - `train.py`  
  - `postprocess.ipynb`
- **step_potential/**  
  - `model.py`  
  - `utils.py`  
  - `train.py`  
  - `postprocess.ipynb`
- **double_well/**  
  - `model.py`  
  - `utils.py`  
  - `train.py`  
  - `postprocess.ipynb`
- **configs/**  
  - `default.py`

## Ejecución

Para entrenar cada PINN, entra en la carpeta correspondiente y ejecuta:

```bash
cd <nombre_de_carpeta>
python main.py
