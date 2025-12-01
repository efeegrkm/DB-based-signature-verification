from siamese_api import compare_signatures

if __name__ == "__main__":
    img1 = "C:\\Users\\efegr\\OneDrive\\Belgeler\\PythonProjects\\SignatureAuthentication\\SiameseModel\\sign_data\\RealLifeTest\\e2.jpg"
    img2 = "C:\\Users\\efegr\\OneDrive\\Belgeler\\PythonProjects\\SignatureAuthentication\\SiameseModel\\sign_data\\RealLifeTest\\esra3.jpg"

    result = compare_signatures(img1, img2)

    print(f"Distance: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Decision: {result['decision']}")
    print("Same writer?" , "Evet" if result["same_writer"] else "Hayır")
