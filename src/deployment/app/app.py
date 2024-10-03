import json

import streamlit as st
import requests


def main():
	st.title("Code classification")
	q = st.text_area(
		"Enter the code",
		value="print('Hello, World!')",
		max_chars=16385,
		height=400,
	)
	if st.button("Classify"):
		data = requests.post(
			"http://localhost:8000/classify/",
			data=json.dumps({"code": q})
		)
		st.write(data.text)


if __name__ == "__main__":
	main()
