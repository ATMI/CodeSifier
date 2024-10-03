import json

import streamlit as st
import requests


def main():
	st.title("Code classification")
	q = st.text_input("Enter the code", value="print('Hello, World!')")
	if st.button("Classify"):
		data = requests.post(
			"https://localhost:8000/classify/",
			json=json.dumps({"code": q})
		)
		st.write(data.text)


if __name__ == "__main__":
	main()
