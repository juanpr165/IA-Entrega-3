from src.schemas import RAGResponse


def test_pregunta_respondible_tiene_fuente():
    """Caso esperado: la respuesta procede del corpus recuperado."""
    response = RAGResponse(
        respuesta="El préstamo ordinario dura catorce días corridos.",
        referencias=["01_prestamos.md"],
    )
    assert "catorce días" in response.respuesta
    assert response.referencias == ["01_prestamos.md"]


def test_pregunta_trampa_declara_desconocimiento():
    """Contrato del prompt para información que no pertenece al corpus."""
    response = RAGResponse(respuesta="No lo sé", referencias=[])
    assert response.respuesta == "No lo sé"
    assert response.referencias == []


def test_referencias_se_deduplican():
    response = RAGResponse(respuesta="Dato respaldado.", referencias=["a.md", "a.md"])
    assert response.referencias == ["a.md"]
