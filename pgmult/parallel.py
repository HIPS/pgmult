

model, args = None, None


def resample_standard_documents(idx,niter):
    def resample(w,z,theta):
        model.add_data(w)
        d = model.documents.pop()
        d.z, d.theta = z, theta
        for _ in range(niter):
            d.resample
        return d.z, d.theta
    return [resample(w,z,theta) for w,z,theta in args[idx]]


def resample_logistic_documents(idx, niter):
    def resample(w,z,omega,psi):
        model.add_data(w)
        d = model.documents.pop()
        d.z, d.omega, d.psi = z, omega, psi
        for _ in range(niter):
            d.resample
        return d.z, d.omega, d.psi
    return [resample(w,z,omega,psi) for w,z,omega,psi in args[idx]]


