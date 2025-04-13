#include "filedialog.h"

#include <ShObjIdl.h>
#include <codecvt>

static std::string OpenFilePrompt(u32 fileTypesCount, COMDLG_FILTERSPEC fileTypes[])
{
    std::string filePathString;
    IFileOpenDialog *pFileOpen;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                          IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (SUCCEEDED(hr))
    {
        hr = pFileOpen->SetFileTypes(fileTypesCount, fileTypes);

        // Show the Open dialog box.
        hr = pFileOpen->Show(NULL);
        // Get the file name from the dialog box.
        if (SUCCEEDED(hr))
        {
            IShellItem *pItem;
            hr = pFileOpen->GetResult(&pItem);
            if (SUCCEEDED(hr))
            {
                PWSTR pszFilePath;
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                if (SUCCEEDED(hr))
                {
                    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                    filePathString = converterX.to_bytes(pszFilePath);
                    CoTaskMemFree(pszFilePath);
                }
                pItem->Release();
            }
        }
        pFileOpen->Release();
    }
    CoUninitialize();
    return filePathString;
}

static std::vector<std::string> OpenMultipleFilesPrompt(u32 fileTypesCount, COMDLG_FILTERSPEC fileTypes[])
{
    std::vector<std::string> filePathStrings;
    IFileOpenDialog *pFileOpen;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                          IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
    if (SUCCEEDED(hr))
    {
        DWORD dwFlags;
        hr = pFileOpen->GetOptions(&dwFlags);
        if (SUCCEEDED(hr)) {
            hr = pFileOpen->SetOptions(dwFlags | FOS_ALLOWMULTISELECT);
        }
        hr = pFileOpen->SetFileTypes(fileTypesCount, fileTypes);

        // Show the Open dialog box.
        hr = pFileOpen->Show(NULL);
        if (SUCCEEDED(hr)) 
        {
            // Get the results from the dialog
            IShellItemArray* pItems;
            hr = pFileOpen->GetResults(&pItems);

            if (SUCCEEDED(hr)) 
            {
                // Get the number of selected items
                DWORD itemCount;
                hr = pItems->GetCount(&itemCount);

                if (SUCCEEDED(hr)) 
                {
                    for (DWORD i = 0; i < itemCount; i++) 
                    {
                        // Get each selected item
                        IShellItem* pItem;
                        hr = pItems->GetItemAt(i, &pItem);

                        if (SUCCEEDED(hr)) 
                        {
                            // Get the display name of the item (i.e., the file path)
                            PWSTR pszFilePath;
                            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                            if (SUCCEEDED(hr)) 
                            {
                                // Output the file path (for example, print it to console)
                                //wprintf(L"Selected file: %s\n", pszFilePath);

                                std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                                filePathStrings.push_back(converterX.to_bytes(pszFilePath));
                                CoTaskMemFree(pszFilePath); // Free the memory allocated for the file path
                            }

                            pItem->Release();  // Release the current IShellItem
                        }
                    }
                }

                pItems->Release();  // Release the IShellItemArray
            }
        }
    }
    CoUninitialize();
    return filePathStrings;
}

static std::string SaveFilePrompt(COMDLG_FILTERSPEC fileTypes[], const wchar_t *defaultExtension)
{
    std::string filePathString;
    IFileSaveDialog* pFileSave;
    HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL,
                                  IID_IFileSaveDialog, reinterpret_cast<void**>(&pFileSave));
    if (SUCCEEDED(hr))
    {
        hr = pFileSave->SetFileTypes(1, fileTypes);
        pFileSave->SetDefaultExtension(defaultExtension);

        // Show the Open dialog box.
        hr = pFileSave->Show(NULL);
        // Get the file name from the dialog box.
        if (SUCCEEDED(hr))
        {
            IShellItem *pItem;
            hr = pFileSave->GetResult(&pItem);
            if (SUCCEEDED(hr))
            {
                PWSTR pszFilePath;
                hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                if (SUCCEEDED(hr))
                {
                    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converterX;
                    filePathString = converterX.to_bytes(pszFilePath);
                    CoTaskMemFree(pszFilePath);
                }
                pItem->Release();
            }
        }
        pFileSave->Release();
    }
    CoUninitialize();
    return filePathString;
}

std::string OpenPNGImageDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"PNG files", L"*.png" } };
    std::string imagepath = OpenFilePrompt(1, fileTypes);
    return imagepath;
}

std::string OpenEditableMapFormatDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Editable map files", L"*.emf" } };
    std::string path = OpenFilePrompt(1, fileTypes);
    return path;
}

std::string SaveEditableMapFormatDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Editable map files", L"*.emf" } };
    std::string path = SaveFilePrompt(fileTypes, L"emf");
    return path;
}

// std::string OpenGameMapDialog()
// {
//     COMDLG_FILTERSPEC fileTypes[] = { { L"Game map files", L"*.map" } };
//     std::string path = OpenFilePrompt(1, fileTypes);
//     return path;
// }

std::string SaveGameMapDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = { { L"Game map files", L"*.map" } };
    std::string path = SaveFilePrompt(fileTypes, L"map");
    return path;
}

std::vector<std::string> OpenImageFilesDialog()
{
    COMDLG_FILTERSPEC fileTypes[] = {
        { L"JPG, PNG, BMP files", L"*.jpg;*.jpeg;*.png;*.bmp" },
    };
    return OpenMultipleFilesPrompt(1, fileTypes);
}
