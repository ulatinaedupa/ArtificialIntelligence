from dotenv import load_dotenv

load_dotenv()


from flask import Flask, request, jsonify
from state_account_agent.graph import state_account_agent

app = Flask(__name__)

@app.route("/feedback", methods=["POST"])
def feedback():
    received = request.json
    data = received.get('data', 'Empty Request')
    # revisar saldo en cuenta 
    # pagar servicios 
    # recordar pagos
    # recomendar ofertas de banco basado en tu ubicacion
    # preguntar por los servicios que ofrece el banco
    # comprar acciones
    # vender acciones
    # recomendar acciones
    
    inputs = {"messages": [("user", data)]}
    response = state_account_agent.invoke(inputs)
    last_message = response.get('messages')[-1]
    return jsonify({'response': last_message.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)