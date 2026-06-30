#include <dlfcn.h>
#include <string>
#include <qualcomm/QNN/QnnInterface.h>
#include <qualcomm/QNN/QnnTypes.h>
struct QnnLoadedBackendContainer {
  void* librarySharedObjectHandle = nullptr;
  const QnnInterface_t* interfaceProviderTable = nullptr;
};
typedef Qnn_ErrorHandle_t (*QnnInterface_getProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
QnnLoadedBackendContainer loadQnnBackendSharedLibrary(const std::string& sharedLibraryPath) {
  QnnLoadedBackendContainer loadedBackend = {nullptr, nullptr};
  void* libraryHandle = dlopen(sharedLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!libraryHandle) {
    return loadedBackend;
  }
  auto getProvidersFunctionPointer = reinterpret_cast<QnnInterface_getProvidersFn_t>(
      dlsym(libraryHandle, "QnnInterface_getProviders"));
  if (!getProvidersFunctionPointer) {
    dlclose(libraryHandle);
    return loadedBackend;
  }
  const QnnInterface_t** interfaceProvidersList = nullptr;
  uint32_t manyProvidersFound = 0;
  Qnn_ErrorHandle_t fetchStatus = getProvidersFunctionPointer(&interfaceProvidersList, &manyProvidersFound);
  if (fetchStatus != QNN_SUCCESS || manyProvidersFound == 0 || !interfaceProvidersList) {
    dlclose(libraryHandle);
    return loadedBackend;
  }
  loadedBackend.librarySharedObjectHandle = libraryHandle;
  loadedBackend.interfaceProviderTable = interfaceProvidersList[0];
  return loadedBackend;
}
void unloadQnnBackendSharedLibrary(QnnLoadedBackendContainer& loadedBackend) {
  if (loadedBackend.librarySharedObjectHandle) {
    dlclose(loadedBackend.librarySharedObjectHandle);
    loadedBackend.librarySharedObjectHandle = nullptr;
    loadedBackend.interfaceProviderTable = nullptr;
  }
}