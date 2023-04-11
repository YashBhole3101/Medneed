
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]